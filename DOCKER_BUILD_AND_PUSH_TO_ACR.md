# Docker Build and Push to ACR

Guide for building Docker images locally and pushing them to Azure Container Registry (ACR).

## Prerequisites

- Docker installed locally
- Azure CLI installed (`az` command)
- Azure account with access to salesaiforecasting ACR

---

## Azure Setup (Do Once)

### Step 1: Authenticate with Azure

Login to your Azure account:

```bash
az login
```

Verify the correct subscription:

```bash
az account show
```

---

### Step 2: Login to Azure Container Registry

Login using Azure CLI (recommended):

```bash
az acr login --name salesaiforecasting
```

Or login manually using Docker:

```bash
az acr credential show --name salesaiforecasting
docker login salesaiforecasting.azurecr.io
```

**Login credentials:**
- Server: `salesaiforecasting.azurecr.io`
- Username: `salesaiforecasting`
- Password: (from `az acr credential show` output)

Expected output: `Login Succeeded`

---

## Backend Build & Push

### Step 1: Build Backend Image

Navigate to backend directory and build:

```bash
cd backend
docker build -t sales_ai_bknd .
cd ..
```

Verify the image was created:

```bash
docker images
```

Expected output:
```
REPOSITORY       TAG       IMAGE ID
sales_ai_bknd    latest    <image-id>
```

---

### Step 2: Test Backend Locally (Optional)

Run the backend container:

```bash
docker run -p 8000:8000 sales_ai_bknd
```

Test the backend:
- Open browser: `http://localhost:8000`
- Verify API endpoints respond
- Test core functionality

Stop the container: `Ctrl + C`

---

### Step 3: Tag Backend Image for ACR

```bash
docker tag sales_ai_bknd:latest salesaiforecasting.azurecr.io/sales_ai_bknd:v1
```

Verify the tagged image:

```bash
docker images
```

Expected output:
```
REPOSITORY                                        TAG       IMAGE ID
salesaiforecasting.azurecr.io/sales_ai_bknd      v1        <image-id>
```

---

### Step 4: Push Backend Image to ACR

Push the image:

```bash
docker push salesaiforecasting.azurecr.io/sales_ai_bknd:v1
```

Verify the image was pushed:

```bash
az acr repository show-tags --name salesaiforecasting --repository sales_ai_bknd
```

---

## Frontend Build & Push

### Step 1: Build Frontend Image

Navigate to frontend directory and build:

```bash
cd frontend
docker build -t sales_ai_fntd .
cd ..
```

Verify the image was created:

```bash
docker images
```

Expected output:
```
REPOSITORY       TAG       IMAGE ID
sales_ai_fntd    latest    <image-id>
```

---

### Step 2: Test Frontend Locally (Optional)

Run the frontend container:

```bash
docker run -p 3000:80 sales_ai_fntd
```

Test the frontend:
- Open browser: `http://localhost:3000`
- Verify UI loads correctly
- Test dashboard and interactions

Stop the container: `Ctrl + C`

---

### Step 3: Tag Frontend Image for ACR

```bash
docker tag sales_ai_fntd:latest salesaiforecasting.azurecr.io/sales_ai_fntd:v1
```

Verify the tagged image:

```bash
docker images
```

Expected output:
```
REPOSITORY                                        TAG       IMAGE ID
salesaiforecasting.azurecr.io/sales_ai_fntd      v1        <image-id>
```

---

### Step 4: Push Frontend Image to ACR

Push the image:

```bash
docker push salesaiforecasting.azurecr.io/sales_ai_fntd:v1
```

Verify the image was pushed:

```bash
az acr repository show-tags --name salesaiforecasting --repository sales_ai_fntd
```

---

## Verify All Images in ACR

List all repositories:

```bash
az acr repository list --name salesaiforecasting --output table
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Login Failed` | Verify credentials with `az acr credential show --name salesaiforecasting` |
| `Access Denied` | Check Azure subscription and ACR permissions |
| `Push Failed` | Ensure image is tagged correctly and ACR login is active |

---

## Quick Reference

### Backend Commands

```bash
# Build
cd backend && docker build -t sales_ai_bknd . && cd ..

# Tag
docker tag sales_ai_bknd:latest salesaiforecasting.azurecr.io/sales_ai_bknd:v1

# Push
docker push salesaiforecasting.azurecr.io/sales_ai_bknd:v1

# Verify
az acr repository show-tags --name salesaiforecasting --repository sales_ai_bknd
```

### Frontend Commands

```bash
# Build (with backend API URL)
cd frontend && docker build --build-arg VITE_API_BASE_URL=https://sales-ai-backend.lemonsand-3ac2a116.westus2.azurecontainerapps.io -t sales_ai_fntd:v2 . && cd ..

# Tag
docker tag sales_ai_fntd:v2 salesaiforecasting.azurecr.io/sales_ai_fntd:v2

# Push
docker push salesaiforecasting.azurecr.io/sales_ai_fntd:v2

# Verify
az acr repository show-tags --name salesaiforecasting --repository sales_ai_fntd
```

**Note:** Replace `VITE_API_BASE_URL` with your actual backend Container Apps URL if different.
