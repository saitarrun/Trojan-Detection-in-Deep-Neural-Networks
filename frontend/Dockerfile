# Production-grade Dockerfile for Next.js
FROM node:20-alpine

WORKDIR /app

# Install dependencies first (for caching)
COPY package.json package-lock.json* ./
RUN npm install

# Copy the rest of the application
COPY . .

# Build the application
# We use dev mode for this playground-style deployment to allow hot reloading if volumes are mounted,
# but for a "Professional UI" we should build. Let's build.
RUN npm run build

# Expose the port Next.js runs on
EXPOSE 3000

# Start the application
CMD ["npm", "run", "start"]
