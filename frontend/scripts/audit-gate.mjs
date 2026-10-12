// npm audit gate: fails on high/critical vulnerabilities, minus documented
// risk exceptions for advisories that have no upstream patch available.
//
// KNOWN_RISK_EXCEPTIONS (review quarterly):
// - GHSA-vfj7-8cjw-p6xm — braces <= 3.0.3 has no patched release upstream
//   (pulled in via @next/eslint-plugin-next -> fast-glob -> micromatch -> braces).
//   Remove from this list once a fixed braces version is published.
import { execSync } from "node:child_process";

const KNOWN_RISK_EXCEPTIONS = new Set(["GHSA-vfj7-8cjw-p6xm"]);

let audit;
try {
  const out = execSync("npm audit --json", {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    maxBuffer: 32 * 1024 * 1024,
  });
  audit = JSON.parse(out);
} catch (e) {
  // npm audit exits non-zero when vulnerabilities are found; the JSON report
  // is still printed to stdout.
  const stdout = e.stdout ? e.stdout.toString() : "{}";
  audit = JSON.parse(stdout);
}

const failures = [];
for (const [name, v] of Object.entries(audit.vulnerabilities || {})) {
  if (v.severity !== "high" && v.severity !== "critical") continue;
  const urls = [];
  for (const via of v.via || []) {
    if (via && typeof via === "object" && via.url) urls.push(via.url);
  }
  // Fail closed: a finding is only excepted when every linked advisory is
  // on the known-risk list.
  const excepted =
    urls.length > 0 &&
    urls.every((u) => [...KNOWN_RISK_EXCEPTIONS].some((g) => u.includes(g)));
  if (!excepted) failures.push(`${name}@${v.range} [${v.severity}] ${urls[0] || ""}`);
}

if (failures.length > 0) {
  console.error("npm audit gate FAILED on high/critical vulnerabilities:");
  for (const f of failures) console.error(" -", f);
  process.exit(1);
}
console.log(
  `npm audit gate passed (risk exceptions: ${[...KNOWN_RISK_EXCEPTIONS].join(", ") || "none"})`
);
