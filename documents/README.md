# Application Architecture

## Overview

**Tracka** is a comprehensive bug tracker application built using **Next.js 15 (App Router)**, **TypeScript**, and **Material UI v7**. It uses **Zustand** for state management and **NextAuth** for authentication. The backend is a separate API (likely Node.js/Express), proxied through Next.js API routes.

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **UI Library**: Material UI v7 (@mui/material, @mui/x-data-grid)
- **State Management**: Zustand
- **Authentication**: NextAuth v4
- **Styling**: Sass
- **Icons**: Material Design Icons
- **Charts**: Chart.js

## Key Directories

- `/src/app`: Next.js App Router pages and layouts.
- `/src/app/_common`: Shared components, hooks, stores, constants, and utilities.
- `/src/app/api`: Next.js API routes (proxy to backend).
- `/public`: Static assets.

## High-Level Architecture Diagram

```mermaid
graph TD
    User[User Browser] -->|HTTPS| NextApp[Next.js App]
    NextApp -->|SSR/SSG| User
    NextApp -->|API Route Proxy| Backend[Backend API]
    Backend -->|Database| DB[(Database)]

    subgraph "Frontend (Next.js)"
        NextApp
        Layouts
        Components
        Stores
    end

    subgraph "Backend"
        Backend
        DB
    end

    Layouts --> Components
    Components --> Stores
```
