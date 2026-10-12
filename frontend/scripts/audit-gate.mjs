// npm audit gate: fails on high/critical vulnerabilities, minus documented
// risk exceptions for advisories that have no upstream patch available.
//
// KNOWN_RISK_EXCEPTIONS (review quarterly):
// - GHSA-vfj7-8cjw-p6xm — braces <= 3.0.3 has no patched release upstream
//   (pulled in via @next/eslint-plugin-next -> fast-glob -> micromatch -> braces).
//   Remove from this list once a fixed braces version is published.
import { execSync } from "node:child_process";

const KNOWN_RISK_EXCEPTIONS = new Set(
  ["GHSA-vfj7-8cjw-p6xm"].map((g) => g.toUpperCase())
);

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

const vulns = audit.vulnerabilities || {};

// Collect every GHSA advisory an entry transitively involves. npm lists each
// package in a vulnerable dependency chain as its own entry; only the root
// entry carries the advisory URL, the rest reference it by package name via
// `via`, so we resolve the chain.
function advisoryIds(name, seen = new Set()) {
  if (seen.has(name)) return new Set();
  seen.add(name);
  const v = vulns[name];
  if (!v) return new Set();
  const ids = new Set();
  for (const via of v.via || []) {
    if (via && typeof via === "object" && via.url) {
      const m = via.url.match(/GHSA-[a-z0-9-]+/i);
      if (m) ids.add(m[0].toUpperCase());
    } else if (typeof via === "string" && vulns[via]) {
      for (const id of advisoryIds(via, seen)) ids.add(id);
    }
  }
  return ids;
}

const failures = [];
for (const name of Object.keys(vulns)) {
  const v = vulns[name];
  if (v.severity !== "high" && v.severity !== "critical") continue;
  const ids = advisoryIds(name);
  // Fail closed: except only when the finding involves at least one advisory
  // and every involved advisory is on the known-risk list.
  const excepted =
    ids.size > 0 && [...ids].every((id) => KNOWN_RISK_EXCEPTIONS.has(id));
  if (!excepted) {
    failures.push(`${name}@${v.range} [${v.severity}] ${[...ids].join(", ")}`);
  }
}

if (failures.length > 0) {
  console.error("npm audit gate FAILED on high/critical vulnerabilities:");
  for (const f of failures) console.error(" -", f);
  process.exit(1);
}
console.log(
  `npm audit gate passed (risk exceptions: ${[...KNOWN_RISK_EXCEPTIONS].join(", ") || "none"})`
);
