# Contributing to AgriPrice-AI

Thank you for your interest in improving AgriPrice-AI, a production-grade machine learning platform for Indian agricultural market intelligence.

## Development Workflow

1. **Fork & Clone**
   ```bash
   git clone https://github.com/rohit8882-code/AgriPrice-AI.git
   cd AgriPrice-AI
   ```

2. **Setup Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   make setup
   ```

3. **Code Style & Standards**
   - Format code using `black` (`make format`).
   - Lint using `ruff` (`make lint`).
   - Never commit model binary files (`*.pkl`), databases (`*.db`), or raw credentials (`.env`).
   - Ensure all data not sourced directly from verified government feeds is tagged `SIMULATED`.

4. **Running Tests**
   ```bash
   make test
   ```

5. **Submitting a Pull Request**
   - Create a feature branch: `git checkout -b feature/your-feature-name`.
   - Ensure all unit and integration tests pass.
   - Commit with descriptive messages.
   - Open a PR against `main`.
