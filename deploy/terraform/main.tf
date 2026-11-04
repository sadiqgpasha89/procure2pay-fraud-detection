# Google Cloud reference deployment. NOT applied or `terraform validate`d in this repository's CI (no GCP project / terraform binary
# was available when it was written) — review the plan before applying. Cost-conscious defaults: scale-to-zero except the gateway and
# scoring service (min 1 for latency), smallest Cloud SQL tier, Gemini Flash-class model, lexical RAG (no embedding spend).
terraform {
  required_version = ">= 1.6"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "us-central1"
}
variable "image_tag" {
  type    = string
  default = "latest"
}
variable "gemini_model" {
  type    = string
  default = "gemini-3.5-flash"
}
variable "audit_retention_days" {
  type    = number
  default = 2555 # ~7 years (SOX)
}
variable "lock_audit_retention" {
  type    = bool
  default = false # set true only after validation: a locked retention policy is irreversible
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "apis" {
  for_each = toset(["run", "artifactregistry", "aiplatform", "secretmanager", "sqladmin", "bigquery", "storage", "cloudbuild", "pubsub"])
  service  = "${each.key}.googleapis.com"
}

resource "google_artifact_registry_repository" "p2p" {
  repository_id = "p2p"
  format        = "DOCKER"
  location      = var.region
  depends_on    = [google_project_service.apis]
}

locals {
  image = "${var.region}-docker.pkg.dev/${var.project_id}/p2p/app:${var.image_tag}"
  # name → { port, min instances, needs Vertex AI, extra env }
  services = {
    "gateway"       = { port = 8100, min = 1, vertex = false, public = true }
    "scoring"       = { port = 8101, min = 1, vertex = false, public = false }
    "case"          = { port = 8102, min = 0, vertex = false, public = true }
    "audit"         = { port = 8103, min = 1, vertex = false, public = false } # single writer ⇒ max 1 instance
    "mcp-erp"       = { port = 8201, min = 0, vertex = false, public = false }
    "mcp-graph"     = { port = 8202, min = 0, vertex = false, public = false }
    "mcp-knowledge" = { port = 8203, min = 0, vertex = false, public = false }
    "graph-analyst" = { port = 8301, min = 0, vertex = true, public = false }
    "doc-auditor"   = { port = 8302, min = 0, vertex = true, public = false }
    "investigator"  = { port = 8300, min = 0, vertex = true, public = false }
  }
  urls = { for k, v in local.services : k => "https://${k}-${data.google_project.this.number}.${var.region}.run.app" }
}

data "google_project" "this" { project_id = var.project_id }

# ── secrets / identities ────────────────────────────────────────────────────────
resource "random_password" "service_token" {
  length  = 48
  special = false
}
resource "google_secret_manager_secret" "service_token" {
  secret_id = "p2p-service-token"
  replication { auto {} }
  depends_on = [google_project_service.apis]
}
resource "google_secret_manager_secret_version" "service_token" {
  secret      = google_secret_manager_secret.service_token.id
  secret_data = random_password.service_token.result
}

resource "google_service_account" "svc" {
  for_each     = local.services
  account_id   = "p2p-${each.key}"
  display_name = "p2p ${each.key}"
}
resource "google_secret_manager_secret_iam_member" "token_access" {
  for_each  = local.services
  secret_id = google_secret_manager_secret.service_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.svc[each.key].email}"
}
resource "google_project_iam_member" "vertex" {
  for_each = { for k, v in local.services : k => v if v.vertex }
  project  = var.project_id
  role     = "roles/aiplatform.user" # Gemini on Vertex AI
  member   = "serviceAccount:${google_service_account.svc[each.key].email}"
}

# ── storage: WORM audit bucket, model/data bucket, BigQuery mirror, Cloud SQL ─────────────────
resource "google_storage_bucket" "audit" {
  name                        = "${var.project_id}-p2p-audit-worm"
  location                    = var.region
  uniform_bucket_level_access = true
  versioning { enabled = true }
  retention_policy {
    retention_period = var.audit_retention_days * 86400
    is_locked        = var.lock_audit_retention
  }
}
resource "google_storage_bucket_iam_member" "audit_writer" {
  bucket = google_storage_bucket.audit.name
  role   = "roles/storage.objectCreator" # create-only: the audit service cannot overwrite or delete
  member = "serviceAccount:${google_service_account.svc["audit"].email}"
}
resource "google_storage_bucket" "data" { # warehouse snapshot, model bundle, RAG index (read-only mounts)
  name                        = "${var.project_id}-p2p-data"
  location                    = var.region
  uniform_bucket_level_access = true
}
resource "google_storage_bucket_iam_member" "data_reader" {
  for_each = toset(["scoring", "mcp-erp", "mcp-graph", "mcp-knowledge"])
  bucket   = google_storage_bucket.data.name
  role     = "roles/storage.objectViewer"
  member   = "serviceAccount:${google_service_account.svc[each.key].email}"
}
resource "google_bigquery_dataset" "audit" {
  dataset_id = "p2p_audit"
  location   = var.region
}
resource "google_bigquery_table" "ledger" {
  dataset_id          = google_bigquery_dataset.audit.dataset_id
  table_id            = "ledger"
  deletion_protection = true
  time_partitioning { type = "DAY" }
  schema = jsonencode([
    { name = "index", type = "INTEGER", mode = "REQUIRED" }, { name = "timestamp", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "event_type", type = "STRING" }, { name = "record_id", type = "STRING" }, { name = "actor", type = "STRING" },
    { name = "payload", type = "STRING" }, { name = "previous_hash", type = "STRING" }, { name = "hash", type = "STRING" },
  ])
}
resource "google_bigquery_table_iam_member" "audit_bq" {
  dataset_id = google_bigquery_dataset.audit.dataset_id
  table_id   = google_bigquery_table.ledger.table_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.svc["audit"].email}"
}

resource "google_sql_database_instance" "state" { # case + decision stores
  name                = "p2p-state"
  database_version    = "POSTGRES_16"
  region              = var.region
  deletion_protection = true
  settings {
    tier = "db-f1-micro"
    backup_configuration { enabled = true }
  }
  depends_on = [google_project_service.apis]
}
resource "google_sql_database" "p2p" {
  name     = "p2p"
  instance = google_sql_database_instance.state.name
}
resource "google_project_iam_member" "sql_client" {
  for_each = toset(["case", "gateway"])
  project  = var.project_id
  role     = "roles/cloudsql.client"
  member   = "serviceAccount:${google_service_account.svc[each.key].email}"
}
resource "random_password" "db" {
  length  = 32
  special = false
}
resource "google_sql_user" "p2p" {
  name     = "p2p"
  instance = google_sql_database_instance.state.name
  password = random_password.db.result
}
resource "google_secret_manager_secret" "state_dsn" { # full SQLAlchemy DSN, injected as CASE_DB_URL / DECISION_DB_URL
  secret_id = "p2p-state-dsn"
  replication { auto {} }
}
resource "google_secret_manager_secret_version" "state_dsn" {
  secret      = google_secret_manager_secret.state_dsn.id
  secret_data = "postgresql+psycopg://p2p:${random_password.db.result}@/p2p?host=/cloudsql/${google_sql_database_instance.state.connection_name}"
}
resource "google_secret_manager_secret_iam_member" "dsn_access" {
  for_each  = toset(["case", "gateway"])
  secret_id = google_secret_manager_secret.state_dsn.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.svc[each.key].email}"
}

# ── Cloud Run services ──────────────────────────────────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "svc" {
  for_each = local.services
  name     = each.key
  location = var.region
  template {
    service_account = google_service_account.svc[each.key].email
    scaling {
      min_instance_count = each.value.min
      max_instance_count = each.key == "audit" ? 1 : 10
    }
    containers {
      image = local.image
      args  = [each.key]
      ports { container_port = each.value.port }
      resources { limits = { cpu = "1", memory = each.value.vertex ? "1Gi" : "512Mi" } }
      env {
        name  = "PORT"
        value = tostring(each.value.port)
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "LLM_PROVIDER"
        value = "vertex"
      }
      env {
        name  = "GEMINI_MODEL"
        value = var.gemini_model
      }
      env {
        name  = "AUDIT_SINK"
        value = "gcs"
      }
      env {
        name  = "AUDIT_GCS_BUCKET"
        value = google_storage_bucket.audit.name
      }
      env {
        name  = "AUDIT_BQ_TABLE"
        value = "${var.project_id}.${google_bigquery_dataset.audit.dataset_id}.${google_bigquery_table.ledger.table_id}"
      }
      env {
        name  = "DATA_DIR"
        value = "/data"
      }
      dynamic "env" {
        for_each = contains(["case", "gateway"], each.key) ? ["CASE_DB_URL", "DECISION_DB_URL"] : []
        content {
          name = env.value
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.state_dsn.secret_id
              version = "latest"
            }
          }
        }
      }
      dynamic "env" {
        for_each = {
          SCORING_URL = local.urls["scoring"], CASE_URL = local.urls["case"], AUDIT_URL = local.urls["audit"],
          MCP_ERP_URL = "${local.urls["mcp-erp"]}/mcp", MCP_GRAPH_URL = "${local.urls["mcp-graph"]}/mcp", MCP_KNOWLEDGE_URL = "${local.urls["mcp-knowledge"]}/mcp",
          GRAPH_ANALYST_URL = local.urls["graph-analyst"], DOC_AUDITOR_URL = local.urls["doc-auditor"], INVESTIGATOR_URL = local.urls["investigator"],
        }
        content {
          name  = env.key
          value = env.value
        }
      }
      env {
        name = "SERVICE_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.service_token.secret_id
            version = "latest"
          }
        }
      }
      dynamic "volume_mounts" {
        for_each = contains(["case", "gateway"], each.key) ? [1] : []
        content {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }
      dynamic "volume_mounts" {
        for_each = contains(["scoring", "mcp-erp", "mcp-graph", "mcp-knowledge"], each.key) ? [1] : []
        content {
          name       = "data"
          mount_path = "/data"
        }
      }
    }
    dynamic "volumes" {
      for_each = contains(["case", "gateway"], each.key) ? [1] : []
      content {
        name = "cloudsql"
        cloud_sql_instance { instances = [google_sql_database_instance.state.connection_name] }
      }
    }
    dynamic "volumes" {
      for_each = contains(["scoring", "mcp-erp", "mcp-graph", "mcp-knowledge"], each.key) ? [1] : []
      content {
        name = "data"
        gcs {
          bucket    = google_storage_bucket.data.name
          read_only = true
        }
      }
    }
  }
  depends_on = [google_secret_manager_secret_version.service_token, google_secret_manager_secret_version.state_dsn]
}

# Services enforce the bearer token themselves (BearerAuthMiddleware), so Cloud Run-level invocation is open.
# Hardening path: ingress=internal + Direct VPC egress + OIDC ID tokens (roles/run.invoker per caller) — see docs/ARCHITECTURE.md §9.
resource "google_cloud_run_v2_service_iam_member" "invoker" {
  for_each = local.services
  name     = google_cloud_run_v2_service.svc[each.key].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "gateway_url" { value = google_cloud_run_v2_service.svc["gateway"].uri }
output "console_url" { value = google_cloud_run_v2_service.svc["case"].uri }
