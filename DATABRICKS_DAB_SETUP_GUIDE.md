# Databricks Asset Bundle (DAB) – Setup Guide

## 1. Install Databricks CLI
### WinGet Installation / Troubleshooting

Check whether WinGet is installed:

```powershell
winget --version
```

If WinGet is not recognized, install/update **App Installer**, which provides WinGet.

### Official WinGet documentation

https://learn.microsoft.com/windows/package-manager/winget/

After installing/updating App Installer:

```powershell
winget --version
```

Then install Databricks CLI:

```powershell
winget install Databricks.DatabricksCLI
```

Verify:

```powershell
databricks --version
```

---

If `databricks` is not recognized, restart the terminal after installation.

### macOS

```bash
brew tap databricks/tap
brew install databricks
```

Verify:

```bash
databricks -v
```

---

## 2. Initialize the Databricks Asset Bundle

Navigate to the required project location:

```bash
cd <project-folder>
```

Initialize the bundle:

```bash
databricks bundle init
```

Select the required bundle template and provide the project name when prompted.

> After the bundle is initialized, clean up the generated files/folders as required for your project.

---

## 3. Configure `databricks.yml`

Update `databricks.yml` with the required project name, workspace host, catalog, and schema.

Example:

```yaml
bundle:
  name: <project-name>

include:
  - resources/*.yml

variables:
  catalog:
    description: The catalog to use
  schema:
    description: The schema to use

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: <databricks-workspace-url>
      root_path: /Workspace/<shared-folder>/.bundle/${workspace.current_user.short_name}/${bundle.name}/${bundle.target}
    variables:
      catalog: <catalog-name>
      schema: <schema-name>
```

---

## 4. Generate the Existing Databricks Job

Use the existing Databricks Job ID:

```bash
databricks bundle generate job --existing-job-id <JOB_ID> --target dev
```

This generates the Job resource YAML under the bundle's `resources` folder.

---

## 5. Validate the Bundle

Run from the DAB project root:

```bash
databricks bundle validate -t dev
```

Expected result:

```text
Validation passed
```

---

## 6. Review Bundle Summary

```bash
databricks bundle summary -t dev
```

Review the resources that will be deployed.

---

## 7. Deploy the Bundle

```bash
databricks bundle deploy -t dev
```

This deploys the bundle resources to the configured Databricks workspace.

---


# Databricks Asset Bundle – Azure DevOps Deployment

## 1. Create Azure DevOps Pipeline

- Create a **Classic Editor** pipeline.
- Connect it to the Git repository containing the DAB project.

DAB location:

```text
Databricks/sales_ai/
├── databricks.yml
├── resources/
└── src/
```

## 2. Install Databricks CLI

Add a **Command Line** task:

```bash
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks -v
```

## 3. Configure Authentication

Create Azure DevOps variables:

```text
DATABRICKS_HOST
DATABRICKS_TOKEN
```

Mark `DATABRICKS_TOKEN` as **secret**.

For Databricks Free Edition, use your Databricks PAT.

For Databricks CLI tasks, add:

```text
DATABRICKS_HOST=$(DATABRICKS_HOST)
DATABRICKS_TOKEN=$(DATABRICKS_TOKEN)
```

## 4. Test Authentication

Add a **Command Line** task:

```bash
databricks current-user me
```

The command should complete successfully.

## 5. Validate Bundle

Add a **Command Line** task:

```bash
cd "$(Build.SourcesDirectory)/Databricks/sales_ai"
databricks bundle validate -t dev
```

Expected:

```text
Validation passed
```

## 6. Deploy Bundle

Add a **Command Line** task:

```bash
cd "$(Build.SourcesDirectory)/Databricks/sales_ai"
databricks bundle deploy -t dev
```

The deployment location is controlled by the `dev` target in `databricks.yml`.

## 7. Final Pipeline

```text
Git Checkout
    ↓
Install Databricks CLI
    ↓
Test Authentication
    ↓
Validate Bundle
    ↓
Deploy Bundle
```

After the deployment is verified, enable the Azure DevOps **CI trigger** so a Git push automatically starts the pipeline.

## Switch DEV Deployment to Production Mode

Update the existing `dev` target in `databricks.yml`:

```yaml
targets:
  dev:
    mode: production
    workspace:
      host: https://dbc-0101d67c-183e.cloud.databricks.com
      root_path: /Workspace/Sales-AI/.bundle/sales_ai/dev

```
## References

- Databricks CLI: https://docs.databricks.com/aws/en/dev-tools/cli/
- Databricks Asset Bundles: https://docs.databricks.com/aws/en/dev-tools/bundles/
- Azure DevOps CI/CD: https://docs.databricks.com/aws/en/dev-tools/ci-cd/azure-devops
