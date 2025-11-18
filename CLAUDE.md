# CLAUDE.md - AI Assistant Guide

This document provides comprehensive guidance for AI assistants working with this codebase.

## Repository Overview

**Status**: Fresh repository
**Purpose**: General-purpose development project
**Git Branch Strategy**: Feature branches with `claude/` prefix

## Repository Structure

```
/
├── .git/           # Git repository metadata
└── CLAUDE.md       # This file - AI assistant guide
```

### Expected Structure (To Be Established)

As the project grows, consider organizing with:
```
/
├── src/            # Source code
├── tests/          # Test files
├── docs/           # Documentation
├── config/         # Configuration files
├── scripts/        # Utility scripts
├── .github/        # GitHub workflows and templates
└── package.json    # Project metadata (if Node.js/npm)
```

## Development Workflow

### Git Workflow

1. **Branch Naming Convention**
   - Feature branches: `claude/claude-md-<session-id>`
   - All branches MUST start with `claude/` prefix
   - Never push directly to `main` or `master`

2. **Commit Guidelines**
   - Write clear, descriptive commit messages
   - Use conventional commit format when possible:
     - `feat:` - New feature
     - `fix:` - Bug fix
     - `docs:` - Documentation changes
     - `refactor:` - Code refactoring
     - `test:` - Adding tests
     - `chore:` - Maintenance tasks

3. **Push Strategy**
   - Always use: `git push -u origin <branch-name>`
   - Verify branch name starts with `claude/`
   - Retry on network failures with exponential backoff (2s, 4s, 8s, 16s)
   - Maximum 4 retry attempts

### Code Quality Standards

#### General Principles

- **Simplicity First**: Prefer simple, readable solutions over clever ones
- **Consistency**: Follow existing patterns in the codebase
- **Documentation**: Comment complex logic and maintain up-to-date docs
- **Testing**: Write tests for new features and bug fixes
- **Security**: Never introduce common vulnerabilities (XSS, SQL injection, etc.)

#### Before Committing

- [ ] Code follows project style guidelines
- [ ] Tests pass (if test suite exists)
- [ ] No unused imports or variables
- [ ] No hardcoded credentials or secrets
- [ ] Documentation updated if needed
- [ ] Code is properly formatted

## AI Assistant Best Practices

### Planning and Execution

1. **Use TodoWrite Tool Frequently**
   - Break down complex tasks into steps
   - Track progress visibly for users
   - Mark todos complete immediately after finishing
   - Keep only one task as `in_progress` at a time

2. **Research Before Implementing**
   - Use Explore agent for understanding codebase structure
   - Read existing code before making changes
   - Understand context before proposing solutions

3. **Parallel Tool Execution**
   - Run independent operations in parallel
   - Example: Multiple file reads, parallel git commands
   - Sequential only when operations depend on each other

### Communication Guidelines

- **Concise**: Keep responses short and focused
- **No Emojis**: Unless explicitly requested by user
- **Professional**: Objective, technical, fact-based
- **References**: Use `file_path:line_number` format for code references
- **Progress**: Update user on significant steps

### Tool Usage Priorities

1. **File Operations**
   - Read tool for viewing files (not `cat`)
   - Edit tool for modifications (not `sed`)
   - Write tool for new files (not `echo >`)

2. **Search Operations**
   - Glob for finding files by pattern
   - Grep for searching file contents
   - Task tool with Explore agent for complex searches

3. **Avoid**
   - Don't use bash `echo` for communication
   - Don't use `find` when Glob is available
   - Don't use `grep` command when Grep tool exists

## Security Guidelines

### What to Allow

- Authorized security testing (with clear context)
- Defensive security implementations
- CTF challenges and educational contexts
- Security research with proper authorization

### What to Refuse

- Destructive techniques without authorization
- DoS attacks or mass targeting
- Supply chain compromise attempts
- Detection evasion for malicious purposes
- Dual-use security tools without clear authorization context

## Technology Stack

### To Be Determined

As the project develops, document:
- Programming language(s)
- Framework(s) and libraries
- Build tools and task runners
- Testing frameworks
- Deployment targets
- CI/CD pipelines

## Common Tasks

### Starting New Development

```bash
# Ensure on correct branch
git checkout -b claude/claude-md-<session-id>

# Make changes
# ... your development work ...

# Stage and commit
git add .
git commit -m "feat: descriptive message"

# Push with retry logic
git push -u origin claude/claude-md-<session-id>
```

### Adding Dependencies

Document here as project type becomes clear:
- Package manager commands
- Dependency update procedures
- Security scanning tools

### Running Tests

Document here once testing is established:
- Test command
- Coverage requirements
- Pre-commit hooks

### Building/Deploying

Document here as deployment is configured:
- Build commands
- Environment setup
- Deployment process

## Project-Specific Conventions

### Code Style

**To be established** - Document as conventions emerge:
- Indentation (spaces vs tabs)
- Line length limits
- Naming conventions
- File organization patterns

### Architecture Patterns

**To be established** - Document as architecture develops:
- Design patterns in use
- Module organization
- Dependency injection approach
- State management

### Testing Conventions

**To be established** - Document as tests are added:
- Test file naming
- Test organization
- Mocking strategies
- Coverage expectations

## Troubleshooting

### Common Issues

#### Git Push Fails with 403

**Cause**: Branch name doesn't match required pattern
**Solution**: Ensure branch starts with `claude/` and includes session ID

#### Merge Conflicts

**Approach**:
1. Fetch latest from main branch
2. Attempt merge or rebase
3. Resolve conflicts carefully
4. Test thoroughly after resolution

#### Network Failures

**Solution**: Implement retry logic with exponential backoff
- Try operation
- If fails, wait 2 seconds and retry
- Double wait time for each subsequent retry (4s, 8s, 16s)
- Maximum 4 attempts

## Maintenance

### Updating This Document

This CLAUDE.md should be updated when:
- New conventions are established
- Technology stack is chosen/changed
- Workflow processes are modified
- Common issues are discovered
- Project structure evolves

Keep this document accurate and current to maximize AI assistant effectiveness.

## Questions and Feedback

For issues or suggestions about AI assistant behavior:
- Report at: https://github.com/anthropics/claude-code/issues
- Use `/help` command in Claude Code CLI

---

**Last Updated**: 2025-11-18
**Repository State**: Initialized (empty)
**Next Steps**: Establish project structure and technology stack
