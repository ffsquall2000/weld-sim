# =============================================================================
# Frontend Dockerfile — Ultrasonic Metal Welding Virtual Simulation Platform
# Multi-stage build: node build -> nginx serve
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build — compile the Vue/Vite application
# ---------------------------------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY package.json package-lock.json* ./

# Install dependencies (clean install for reproducible builds)
RUN npm ci

# Copy the rest of the source code
COPY . .

# Build the production bundle
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Serve — lightweight nginx to serve the SPA
# ---------------------------------------------------------------------------
FROM nginx:alpine AS runtime

LABEL maintainer="weldsim-team"
LABEL description="WeldSim frontend SPA served by nginx"

# Remove default nginx site configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built static files from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
