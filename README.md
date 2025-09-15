# CrowdClue — DevOps Assignment (Hackathon)

**Short description:**
CrowdClue is a small crowdsourced image-clue game. Admin uploads images. Users register/login, view images and submit one clue per image. Admin reviews and accepts clues; accepted contributors earn points (scoreboard). The backend sanitizes uploads (blurs exposed skin/faces if available) and serves images. This repo provides a Docker Compose demo and Kubernetes manifests for k3d/k3s. CI builds images and pushes to GitHub Container Registry (GHCR).

---

## Repo layout

* `backend/` — Flask backend service and Dockerfile
* `frontend/` — Vite + React app, built and served by nginx in production Dockerfile
* `infra/docker-compose.yml` — local demo (Postgres + backend + frontend)
* `k8/` — Kubernetes manifests (secret, postgres, backend, frontend, ingress)
* `.github/workflows/ci.yml` — GitHub Actions build + push workflow
* `tests/run_demo.sh` — demo harness (curl-based) to validate flows

---

## How it works (quick)

1. Admin uploads images (using `x-admin-secret` header). Images are sanitized (faces/skin blurred when detected).
2. Users register/login; JWT token is returned. Users post one clue per image.
3. Admin lists pending clues and accepts them. When accepted, contributor earns points (+10).
4. Scoreboard lists top users.

Default `ADMIN_SECRET` is `letmein` in the demo configuration.

---

## Quick local demo (Docker Compose) — recommended for judge

Prereqs: Docker Desktop (or Docker Engine) and docker-compose.

```bash
# from repo root
cd infra

# build images (optional - docker compose will build them automatically first time)
docker compose build

# bring services up
docker compose up -d

# check backend health
curl http://localhost:3000/health

# open the frontend in browser:
# http://localhost:8080
```

### Demo flow (example commands)

```bash
# register
curl -s -X POST -H "Content-Type: application/json" -d '{"name":"jason","password":"secret"}' http://localhost:3000/api/register

# login -> get token (you need jq or parse JSON)
TOKEN=$(curl -s -X POST -H "Content-Type: application/json" -d '{"name":"jason","password":"secret"}' http://localhost:3000/api/login | jq -r .token)

# upload image as admin (from host)
curl -X POST -H "x-admin-secret: letmein" -F "image=@/path/to/photo.jpg" -F "title=demo" http://localhost:3000/admin/images

# post a clue (user must be logged in)
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"text":"possible clue"}' http://localhost:3000/api/images/1/clues

# admin accept a clue id=1
curl -X POST -H "x-admin-secret: letmein" http://localhost:3000/admin/clues/1/accept

# scoreboard
curl http://localhost:3000/api/scoreboard
```

---

## Optional: k3d / Kubernetes demo

Prereqs: `k3d`, `kubectl`, docker (k3d uses local docker).

1. Create cluster (run in **WSL bash** or Linux shell — not Windows PowerShell; run each command line separately, not as a pasted block with comments):

```bash
k3d cluster create crowdclue --servers 1 --agents 0 -p "8080:80@server[0]" -p "3000:3000@server[0]" -p "5432:5432@server[0]"
```

If that port syntax fails on your environment, create the cluster without `-p` and use `kubectl port-forward` (instructions below).

2. Build images locally and import into k3d:

```bash
# from repo root
docker build -t crowdclue/backend:dev ./backend
docker build -t crowdclue/frontend:dev ./frontend

# import images into k3d so cluster can use them
k3d image import crowdclue/backend:dev -c crowdclue
k3d image import crowdclue/frontend:dev -c crowdclue
```

3. Install ingress-nginx (k3d provider manifest):

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/k3d/deploy.yaml
```

(If that exact URL 404s because versions changed, use the ingress-nginx install instructions from the official repo and choose the provider manifest for k3d.)

4. Add hosts mapping on your host (so `crowdclue.local` resolves):

```
# On Linux/WSL /etc/hosts (add line):
127.0.0.1 crowdclue.local
```

5. Apply k8 manifests:

```bash
kubectl apply -f k8/secret.yaml
kubectl apply -f k8/postgres-deploy.yaml
kubectl apply -f k8/backend-deploy.yaml
kubectl apply -f k8/frontend-deploy.yaml
kubectl apply -f k8/ingress.yaml
```

6. Browse:

```
http://crowdclue.local
```

**Alternative (if `k3d` port mapping fails):** port-forward the services

```bash
# open two terminals (or backgrounded processes)
kubectl port-forward svc/frontend 8080:80
kubectl port-forward svc/backend 3000:3000
# then visit http://localhost:8080
```

---

## How judge can view CI / workflow logs

1. Look at the `Actions` tab in the GitHub repo — each push to `dev`/`main` or PR will run the workflow. Click a run to view build & push logs (backend and frontend image builds).
2. The workflow pushes images to GHCR (`ghcr.io/<owner>/<repo>-backend` and `...-frontend`). The logs show the build steps.

---

## How to push & trigger CI (simple)

```bash
git checkout -b dev
git add .
git commit -m "final: infra, k8 manifests, CI"
git push -u origin dev
```

This push triggers the `build-and-push` workflow in `.github/workflows/ci.yml`.

---

## Handover / What the judge should check

* Run `infra/docker-compose.yml` locally and exercise web UI.
* Look at `k8/` manifests and optionally deploy to k3d or a cluster.
* Open GitHub Actions → check build logs for backend & frontend builds.
* Run `tests/run_demo.sh` for a scripted flow (all example requests).

---

## Security notes (for judges)

* `VITE_ADMIN_SECRET` is only for demo convenience; DO NOT bake secrets into client builds in real products.
* k8 manifests use `emptyDir` for Postgres data for demo purposes only — not for production.
