# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in NapRobo, please follow these guidelines:

### What to Report

- Buffer overflows or memory safety issues
- Path traversal vulnerabilities
- Code injection risks
- Exposed credentials or secrets
- Unsafe deserialization
- Denial of service vectors
- Information disclosure

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead:
1. Email the maintainer at: [Create a security contact email]
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity
  - Critical: Immediate fix and release
  - High: Fix within 1 week
  - Medium: Fix within 2 weeks
  - Low: Fix in next release

### Security Best Practices

When using NapRobo:

1. **Never commit secrets** to the repository
2. **Validate all inputs** from external sources
3. **Use secure file paths** - avoid user-controlled paths without validation
4. **Keep dependencies updated** - run `pip audit` regularly
5. **Review checkpoint files** before loading from untrusted sources
6. **Use environment variables** for configuration, not hardcoded values

### Security Features

NapRobo includes several security measures:

- **Path validation**: Prevents path traversal attacks
- **Safe JSON loading**: Validates structure before use
- **Error handling**: Prevents information disclosure
- **Input sanitization**: Cleans user inputs
- **Secure file permissions**: Restricts access to sensitive files

### Known Security Limitations

- The online learning module uses simulated search results (not production-ready)
- Checkpoint files are not encrypted (contains model weights)
- No authentication for multi-user deployments

### Security Updates

Security updates are released as patch versions (e.g., 1.0.1). Subscribe to releases to be notified of updates.

## Security Architecture

```
┌─────────────────────────────────────────┐
│         Security Layers                 │
├─────────────────────────────────────────┤
│  Input Validation                       │
│  ├─ Path sanitization                   │
│  ├─ JSON structure validation           │
│  └─ Type checking                       │
├─────────────────────────────────────────┤
│  Safe Operations                        │
│  ├─ Validated file paths                │
│  ├─ Secure JSON I/O                     │
│  └─ Controlled error messages           │
├─────────────────────────────────────────┤
│  Data Protection                        │
│  ├─ No hardcoded secrets                │
│  ├─ Restricted file permissions         │
│  └─ Safe logging (no sensitive data)    │
└─────────────────────────────────────────┘
```

## Contact

For security concerns, contact: [Add your security email]

For general questions, use GitHub Issues.
