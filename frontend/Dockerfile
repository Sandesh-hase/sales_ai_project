FROM node:22-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Vite inlines VITE_* variables into the static bundle at BUILD time, so this
# must be a build arg (not just a runtime env var) to take effect.
ARG VITE_API_BASE_URL=https://salesai-backend.orangewave-aeb38b8d.westus2.azurecontainerapps.io
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
