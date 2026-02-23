# Project Instructions

## Line Length

Keep all lines ≤ 80 columns. `rustfmt.toml` enforces this for code via
`max_width = 80`.

For things rustfmt cannot wrap (comments, string literals, `#[derive(...)]`):
- **Comments:** Wrap manually at a word boundary before column 80.
- **String literals:** Use the escaped-newline trick (`\` at end of line strips
  the newline and leading whitespace on the next line).
- **Long `#[derive(...)]`:** Stable `rustfmt` does not wrap derive
  item lists and merges split `#[derive]` attributes back into one.
  No workaround exists on stable; slightly-over lines are accepted.

Only exceed 80 columns when there is no syntactically valid way to break the
line (e.g., a single token or URL that is itself longer than 80 chars).
