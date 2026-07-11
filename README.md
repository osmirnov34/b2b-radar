# b2b-radar

Automated B2B startup idea discovery from public discussions (YouTube, Reddit, forums, sites)

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/osmirnov34/b2b-radar
cd b2b-radar
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

## Development

### Docker Setup

This project uses Docker for consistent development and deployment environments.

**Build the image:**
```bash
docker-compose build
```

**Run the application:**
```bash
docker-compose up
```

**Run with interactive mode:**
```bash
docker-compose run --rm app bash
```

### Code Quality

This project uses **pre-commit** hooks to automatically lint and format code before commits.

**Installed hooks:**
- **Ruff**: Fast Python linter and formatter (with auto-fix enabled)

**Git hooks run automatically on `git commit`.** To run manually:

```bash
# Run all hooks
pre-commit run --all-files

# Update hook definitions
pre-commit autoupdate
```

**Bypass hooks (not recommended):**
```bash
git commit --no-verify
```
