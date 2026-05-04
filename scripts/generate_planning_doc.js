/**
 * Generates PAARAL_Roadmap_Budget_2026.xlsx
 * Run: node scripts/generate_planning_doc.js
 * Output: docs/PAARAL_Roadmap_Budget_2026.xlsx
 *
 * Budget columns: Item | Unit Cost | Specs | Qty | Amount | Notes | Purpose
 * Pilot scaled to 1,000,000 learners.
 */

import ExcelJS from 'exceljs';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const OUT_DIR  = path.join(__dirname, '..', 'docs');
const OUT_FILE = path.join(OUT_DIR, 'PAARAL_Roadmap_Budget_2026.xlsx');

if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

// ── Colors ───────────────────────────────────────────────────────────────────
const NAVY      = 'FF1E3A5F';
const DEPEDBLUE = 'FF2F5496';
const LIGHTBLUE = 'FFD9E1F2';
const ROWALT    = 'FFEEF3FB';
const WHITE     = 'FFFFFFFF';
const DARKGREEN = 'FF375623';
const TEXTDARK  = 'FF000000';
const TEXTWHITE = 'FFFFFFFF';
const TEXTGREY  = 'FF595959';
const LINKBLUE  = 'FF0563C1';
const TOTALROW  = 'FFBDD7EE';
const RECROW    = 'FFFFE699';

// ── Helpers ───────────────────────────────────────────────────────────────────
function merge(ws, r1, c1, r2, c2) { ws.mergeCells(r1, c1, r2, c2); }

function sectionTitle(ws, text, numCols, fillColor = NAVY, textColor = TEXTWHITE) {
  const row = ws.addRow([text]);
  row.height = 24;
  merge(ws, row.number, 1, row.number, numCols);
  const cell = row.getCell(1);
  cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: fillColor } };
  cell.font      = { bold: true, size: 12, color: { argb: textColor } };
  cell.alignment = { vertical: 'middle', horizontal: 'left', indent: 1 };
  return row;
}

function subSectionTitle(ws, text, numCols) {
  return sectionTitle(ws, text, numCols, LIGHTBLUE, TEXTDARK);
}

function colHeaders(ws, headers, numCols, fillColor = DEPEDBLUE) {
  const row = ws.addRow(headers);
  row.height = 18;
  for (let i = 1; i <= numCols; i++) {
    const cell = row.getCell(i);
    cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: fillColor } };
    cell.font      = { bold: true, size: 10, color: { argb: TEXTWHITE } };
    cell.alignment = { vertical: 'middle', wrapText: true };
  }
  return row;
}

function applyRowStyle(row, numCols, alt, bold = false, fillOverride = null) {
  const fill = fillOverride || (alt ? ROWALT : WHITE);
  for (let i = 1; i <= numCols; i++) {
    const cell = row.getCell(i);
    cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: fill } };
    cell.font      = { bold, size: 10 };
    cell.alignment = { vertical: 'top', wrapText: true };
  }
}

function blank(ws) { ws.addRow([]).height = 8; }

function dataRow(ws, values, numCols, alt, bold = false, fillOverride = null) {
  const row = ws.addRow(values);
  applyRowStyle(row, numCols, alt, bold, fillOverride);
  return row;
}

// ════════════════════════════════════════════════════════════════════════════
// SHEET 1: ROADMAP  (unchanged)
// ════════════════════════════════════════════════════════════════════════════
const wb = new ExcelJS.Workbook();
wb.creator = 'E-CAIR';
wb.created = new Date();

const ws1 = wb.addWorksheet('Roadmap', { views: [{ showGridLines: false }] });
const NC1 = 5;
ws1.columns = [
  { key: 'a', width: 36 },
  { key: 'b', width: 22 },
  { key: 'c', width: 22 },
  { key: 'd', width: 58 },
  { key: 'e', width: 42 },
];

{
  const r = ws1.addRow(['PAARAL — Production Roadmap']);
  r.height = 32;
  merge(ws1, r.number, 1, r.number, NC1);
  r.getCell(1).font      = { bold: true, size: 16, color: { argb: NAVY } };
  r.getCell(1).alignment = { vertical: 'middle' };
}
{
  const r = ws1.addRow(['E-CAIR for the Government Assistance and Subsidies Service (GASS), Department of Education  |  Version 1.0  |  May 2026']);
  r.height = 16;
  merge(ws1, r.number, 1, r.number, NC1);
  r.getCell(1).font = { italic: true, size: 9, color: { argb: TEXTGREY } };
}
blank(ws1);

sectionTitle(ws1, '  THREE VIEWS: WHAT IS NEEDED AND WHEN', NC1);
colHeaders(ws1, ['View', 'Current Status', 'Needed for Jan 2027 Pilot?', 'Target Timeline', 'Rationale'], NC1);

[
  ['Student View',          'Mockup complete (~100% UI)', 'YES — Required',        'January 2027',            'The pilot is student-facing; this is the primary deliverable.'],
  ['School Portal (basic)', 'Not started',                'YES — Required',        'January 2027',            'Schools need to see applications and confirm placements. Without this, pilot coordination is unmanageable.'],
  ['DepEd Planning View',   'Mockup ~90% done',           'NO — Deferred to 2028', 'National Rollout (2028)', 'Requires real BigQuery data at national scale. GASS monitors pilot via aggregate reports and CSV exports.'],
].forEach((v, i) => {
  const row = ws1.addRow(v);
  row.height = 44;
  applyRowStyle(row, NC1, i % 2 === 0);
  row.getCell(1).font = { bold: true, size: 10 };
  const c3 = row.getCell(3);
  if (v[2].startsWith('YES')) {
    c3.font = { bold: true, size: 10, color: { argb: DARKGREEN } };
    c3.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE2EFDA' } };
  } else {
    c3.font = { bold: true, size: 10, color: { argb: '7F6000' } };
    c3.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFF2CC' } };
  }
});

blank(ws1); blank(ws1);
sectionTitle(ws1, '  PRODUCTION ROADMAP  (May 2026 → 2028)', NC1);
{
  const note = ws1.addRow(['⚠️  CRITICAL: GASS FY 2027 budget proposal to DBM is typically due July – August 2026. Phase 0 milestone must be reached before then.']);
  note.height = 18;
  merge(ws1, note.number, 1, note.number, NC1);
  note.getCell(1).fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFF2CC' } };
  note.getCell(1).font      = { bold: true, size: 10, color: { argb: '7F6000' } };
  note.getCell(1).alignment = { vertical: 'middle', indent: 1 };
}
colHeaders(ws1, ['Phase', 'Period', 'Owner', 'Key Activities', 'Milestone / Go-Live Signal'], NC1);

[
  { phase: 'Phase 0 — Mockup Finalization & Sign-Off', period: 'May – June 2026', owner: 'E-CAIR',
    activities: '• Complete DepEd Planning View mockup (final chunk: deployment + edge cases)\n• Design school portal wireframes\n• Stakeholder demo to GASS — sign off on all three view concepts\n• Jointly define pilot scope with GASS: target divisions, school count, learner cohort\n• Produce budget estimates for DBM submission ← THIS DOCUMENT',
    milestone: 'GASS formally endorses UI/UX approach and confirms pilot scope.' },
  { phase: 'Phase 1 — Production Build: Student View + School Portal', period: 'July – September 2026', owner: 'E-CAIR',
    activities: '• Real LRN validation via DepEd EBEIS API (or agreed data-sharing mechanism)\n• School/slot data ingestion from GASS records\n• Persistent wishlist and submission backend\n• School portal: application inbox, accept/waitlist/decline controls, slot dashboard, CSV export\n• Staging environment on GCP asia-southeast1 (Firebase + Cloud Run + Cloud SQL)',
    milestone: 'End-to-end flow functional in staging: student submits → school sees application → school confirms.' },
  { phase: 'Phase 2 — Internal Pilot', period: 'October 2026', owner: 'E-CAIR + GASS',
    activities: '• Deploy to E-CAIR team + select GASS staff + invited school admin contacts\n• End-to-end testing with pilot school data (real or anonymized)\n• Collect structured usability feedback (forms + interviews)\n• Iterate rapidly (2–3 sprint cycles)\n• Finalize pilot school list, learner eligibility criteria, communication plan',
    milestone: 'Internal sign-off that system is ready for external pilot. Go/no-go decision.' },
  { phase: 'Phase 3 — Pilot Preparation', period: 'November – December 2026', owner: 'E-CAIR + GASS',
    activities: '• Onboard pilot schools (real slot data, school admin accounts created)\n• Student-facing outreach (DepEd information drive to eligible Grade 6 learners)\n• School admin training (live sessions + recorded walkthroughs)\n• Production environment setup (separate from staging)\n• GASS monitoring dashboard (submission count, slot fill rate, participation rate)\n• Helpdesk protocol finalized (who handles school vs. student issues)',
    milestone: 'Pilot schools confirmed and onboarded; students notified; system in production.' },
  { phase: 'Phase 4 — January 2027 Pilot  ★', period: 'January – June 2027', owner: 'GASS-led, E-CAIR support',
    activities: '• Student-facing enrollment window opens (aligned with Grade 6 SY transition timeline)\n• Real-time monitoring by E-CAIR + GASS\n• E-CAIR provides break-fix support under soft deployment SLA\n• Post-pilot data collection: slot fill rates, rank distribution, school participation, student feedback',
    milestone: 'Pilot completes. Data collected for sustainability plan. Go/no-go for national rollout.' },
  { phase: 'Phase 5 — Infrastructure Turnover + Sustainability Plan', period: 'June – December 2027', owner: 'E-CAIR → GASS / DepEd',
    activities: '• E-CAIR produces sustainability plan (system architecture, operational runbook, vendor contracts)\n• DepEd ICTS or contracted vendor provisions national infrastructure (DICT cloud or GCP)\n• Codebase, data, and domain transferred to DepEd ownership\n• E-CAIR provides knowledge transfer under paid support contract\n• DepEd Planning View development begins (BigQuery integration + real ILP optimizer)',
    milestone: 'Full ownership transfer complete. National rollout planning underway.' },
  { phase: 'Phase 6 — National Rollout', period: '2028 onward', owner: 'DepEd / GASS',
    activities: '• National student view rollout (all ESC-eligible Grade 6 learners)\n• Full school portal (all ESC-accredited private JHS)\n• DepEd Planning View in production (real ILP optimization on actual student flow data)\n• BigQuery analytics pipeline live',
    milestone: 'Full national deployment. PAARAL operational at scale.' },
].forEach((p, i) => {
  const lines = (p.activities.match(/\n/g) || []).length + 1;
  const row   = ws1.addRow([p.phase, p.period, p.owner, p.activities, p.milestone]);
  row.height  = Math.max(72, lines * 15 + 10);
  applyRowStyle(row, NC1, i % 2 === 0);
  row.getCell(1).font = { bold: true, size: 10 };
  row.getCell(5).font = { italic: true, size: 10 };
  if (p.phase.includes('★')) {
    for (let c = 1; c <= NC1; c++) row.getCell(c).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE2EFDA' } };
    row.getCell(1).font = { bold: true, size: 10, color: { argb: DARKGREEN } };
  }
});

// ════════════════════════════════════════════════════════════════════════════
// SHEET 2: BUDGET
// Columns: Item | Unit Cost | Specs | Qty | Amount | Notes | Purpose
// ════════════════════════════════════════════════════════════════════════════
const ws2 = wb.addWorksheet('Budget', { views: [{ showGridLines: false }] });
const NC2 = 7;
ws2.columns = [
  { key: 'a', width: 30 },   // Item
  { key: 'b', width: 15 },   // Unit Cost
  { key: 'c', width: 38 },   // Specs
  { key: 'd', width: 9  },   // Qty
  { key: 'e', width: 15 },   // Amount
  { key: 'f', width: 30 },   // Notes
  { key: 'g', width: 38 },   // Purpose
];

// Master title
{
  const r = ws2.addRow(['PAARAL — Budget Breakdown for Stakeholder Planning']);
  r.height = 32;
  merge(ws2, r.number, 1, r.number, NC2);
  r.getCell(1).font      = { bold: true, size: 16, color: { argb: NAVY } };
  r.getCell(1).alignment = { vertical: 'middle' };
}
{
  const r = ws2.addRow(['Prepared by E-CAIR for GASS, DepEd Central Office  |  May 2026  |  Pilot scaled to 1,000,000 learners  |  Exchange rate: ₱62 / USD (BSP reference May 1, 2026: ₱61.42 / USD)']);
  r.height = 16;
  merge(ws2, r.number, 1, r.number, NC2);
  r.getCell(1).font = { italic: true, size: 9, color: { argb: TEXTGREY } };
}
blank(ws2);

// ── ASSUMPTIONS ───────────────────────────────────────────────────────────────
sectionTitle(ws2, '  ASSUMPTIONS', NC2);
colHeaders(ws2, ['Parameter', 'Value', 'Description', '', '', 'Notes', ''], NC2);
merge(ws2, ws2.lastRow.number, 3, ws2.lastRow.number, 5);
merge(ws2, ws2.lastRow.number, 7, ws2.lastRow.number, 7);

[
  ['Exchange rate',   '₱62 / USD',                             'Conservative planning rate for 2027 appropriation.',                                                              '',  '',  'BSP reference May 1, 2026: ₱61.42/USD. Slight weakening factored in.', ''],
  ['GCP Region',      'asia-southeast1',                        'Singapore — nearest GCP region to the Philippines.',                                                             '',  '',  'Cloud Run free tier applies in US regions only. Not applicable here.', ''],
  ['Architecture',    'Firebase + Cloud Run + Cloud SQL + BQ', 'Firebase Hosting (frontend), Cloud Run (API), Cloud SQL PostgreSQL (database), BigQuery (analytics), Firebase Auth.', '', '', 'Single GCP account. Shared IAM, billing, no cross-cloud networking costs.', ''],
  ['Pilot Scale',     '1,000,000 learners',                    '~33K learners/day avg; ~165K on peak deadline day; ~15M API requests/month during 30-day enrollment window.',    '',  '',  'Final scope confirmed jointly by E-CAIR and GASS in Phase 0.', ''],
  ['Student Accounts','None (LRN-based, stateless)',           'No Firebase Auth account created per learner. Students identify via LRN; submissions stored in Cloud SQL keyed by LRN. Only school admins (~100–500 users) have accounts.', '', '', 'Adding per-learner Firebase Auth accounts would cost ~₱324,000/month during enrollment (950K MAU × $0.0055). Architecture decision confirmed for pilot.', ''],
  ['DBM Deadline',    'July – August 2026',                    '⚠️ FY 2027 budget proposals to DBM typically due this window. This document supports that submission.',           '',  '',  'GASS must finalize and submit estimates within ~10 weeks of this document.', ''],
  ['Concurrency Limit (2027 Pilot)', 'Max ~20,000–30,000 simultaneous submitting users', '⚠️ The 2027 pilot infrastructure (Cloud SQL 2 vCPU + PgBouncer) handles ~2,000–3,000 API calls/second sustained. This translates to ~20,000–30,000 users actively submitting at the exact same second. All 1M submissions can be processed if spread over at least 2 hours of active enrollment. The system does NOT support all 1M learners submitting simultaneously.', '', '', 'Mitigation: stagger enrollment access by DepEd division or region. Avoid a single national deadline that creates simultaneous pressure. Set rolling deadlines (e.g., Region I: Jan 15, NCR: Jan 17, Region IV-A: Jan 19). The 2028 national rollout requires a database upgrade (Cloud Spanner or Cloud SQL 16+ vCPU) to remove this constraint.', ''],
].forEach((a, i) => {
  const row = ws2.addRow(a);
  row.height = 22;
  merge(ws2, row.number, 3, row.number, 5);
  applyRowStyle(row, NC2, i % 2 === 0);
  row.getCell(1).font = { bold: true, size: 10 };
  if (a[0] === 'DBM Deadline' || a[0] === 'Student Accounts' || a[0].startsWith('Concurrency')) {
    for (let c = 1; c <= NC2; c++) row.getCell(c).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFFFF2CC' } };
    row.getCell(1).font = { bold: true, size: 10, color: { argb: '7F6000' } };
  }
});

blank(ws2); blank(ws2);

// ── 2026: E-CAIR SOFT DEPLOYMENT ─────────────────────────────────────────────
sectionTitle(ws2, '  2026 — E-CAIR SOFT DEPLOYMENT  (No cost to GASS)', NC2);
{
  const note = ws2.addRow(['E-CAIR shoulders all development and hosting costs as an R&D and partnership investment. GASS cost = ₱0 for 2026.']);
  note.height = 18;
  merge(ws2, note.number, 1, note.number, NC2);
  note.getCell(1).fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFE2EFDA' } };
  note.getCell(1).font      = { size: 10, color: { argb: DARKGREEN } };
  note.getCell(1).alignment = { vertical: 'middle', indent: 1 };
}
colHeaders(ws2, ['Cloud Compute and Storage Services', 'Unit Cost', 'Specs', 'Qty', 'Amount', 'Notes', 'Purpose'], NC2);

[
  ['Developer time (Paula + team)',
   '—', 'Full-stack development: React frontend, Python/Node API, Cloud SQL schema, GeoJSON data layer, school portal.', '—', 'Internal',
   'E-CAIR R&D investment. Not charged to GASS.',
   'Builds the production Student View, School Portal, backend API, and data pipeline — the core deliverable of the soft deployment commitment.'],

  ['Firebase Hosting — staging',
   '₱0/month', 'Static hosting; <100 MB/day transfer; 10 GB storage; Firebase free tier.', '12', '₱0',
   'Within 360 MB/day free tier at staging traffic.',
   'Hosts the staging environment where E-CAIR tests features and GASS previews the system before production launch.'],

  ['Cloud Run — staging backend API',
   '₱0–310/month', 'asia-southeast1; <10K req/month (dev traffic); single instance; 0.25 vCPU; 256 MB RAM.', '12', '₱0–3,720',
   'Minimal staging traffic; near free tier levels.',
   'Runs the backend API in staging for integration testing and GASS demonstrations. Mirrors production architecture at development traffic levels.'],

  ['Cloud SQL for PostgreSQL — staging',
   '₱1,860/month', '1 vCPU, 3.75 GB RAM; 10 GB SSD ($0.222/GB/month); asia-southeast1; no high-availability.', '12', '₱22,320',
   'Smallest production-grade instance. No free tier for dedicated Cloud SQL.',
   'Stores test and synthetic data during development. Allows E-CAIR to validate database schema, queries, and data integrity before the January 2027 pilot.'],

  ['Cloud Storage — static assets, GeoJSON',
   '₱62/month', 'Standard storage; <5 GB; $0.020/GB/month; asia-southeast1.', '12', '₱744',
   'School coordinate data, GeoJSON boundary files, synthetic datasets.',
   'Hosts school geographic data and map assets used by both the API and the frontend during development and staging.'],

  ['Firebase Authentication',
   '₱0/month', 'Free tier: 50,000 MAU/month. Staging school admin accounts only.', '12', '₱0',
   'Within 50,000 MAU free tier.',
   'Manages school admin accounts in the staging environment for internal testing and GASS demonstrations.'],

  ['Domain registration',
   '₱1,500/year', '.ph domain (dotPH registry) or ecair.ph subdomain. SSL/TLS via Firebase Hosting (no extra cost).', '1', '₱1,500',
   'paaral.ph or paaral.ecair.ph.',
   'Establishes the public URL for PAARAL. Required before the pilot goes live — students and schools need a stable, recognizable address to access the system.'],
].forEach((row, i) => { dataRow(ws2, row, NC2, i % 2 === 0).height = 36; });

{
  const r = dataRow(ws2, ['TOTAL GASS COST FOR 2026', '', '', '', '₱0', 'E-CAIR investment — all costs absorbed by E-CAIR.', ''], NC2, false, true, DARKGREEN);
  r.height = 20;
  for (let c = 1; c <= NC2; c++) r.getCell(c).font = { bold: true, size: 10, color: { argb: TEXTWHITE } };
}
{
  const r = ws2.addRow(['E-CAIR out-of-pocket for 2026: ~₱26,000 – ₱30,000 in GCP costs + developer time (internal).']);
  merge(ws2, r.number, 1, r.number, NC2);
  r.height = 16;
  r.getCell(1).font      = { italic: true, size: 9, color: { argb: TEXTGREY } };
  r.getCell(1).alignment = { vertical: 'middle', indent: 1 };
}

blank(ws2); blank(ws2);

// ── 2027: GASS PILOT BUDGET ───────────────────────────────────────────────────
sectionTitle(ws2, '  2027 — GASS PILOT BUDGET  (FY 2027 DBM Proposal — MOOE)  |  Scaled to 1,000,000 Learners', NC2);
{
  const note = ws2.addRow(['Budget Category: MOOE. Enrollment window: 30 days (January 2027). Peak: ~165K learners/day; ~15M API requests/month. Unit Cost = average monthly PHP. Qty = 12 months.']);
  merge(ws2, note.number, 1, note.number, NC2);
  note.height = 18;
  note.getCell(1).font      = { italic: true, size: 10 };
  note.getCell(1).alignment = { vertical: 'middle', indent: 1 };
}
blank(ws2);

// A. Cloud Compute and Storage Services
subSectionTitle(ws2, '  A. Cloud Compute and Storage Services  (Google Cloud Platform, asia-southeast1)', NC2);
{
  const note = ws2.addRow(['Cloud Run free tier (2M req/month) applies in US regions only — does NOT apply in asia-southeast1. Firebase Hosting and Auth free tiers apply globally. Enrollment window: 90 days (Jan–Mar 2027).']);
  merge(ws2, note.number, 1, note.number, NC2);
  note.height = 16;
  note.getCell(1).font      = { size: 9, color: { argb: TEXTGREY } };
  note.getCell(1).alignment = { vertical: 'middle', indent: 1 };
}
colHeaders(ws2, ['Cloud Compute and Storage Services', 'Unit Cost', 'Specs', 'Qty', 'Amount', 'Notes', 'Purpose'], NC2);

[
  ['Firebase Hosting  (Student View)',
   '₱55,800/enrollment',
   'Global CDN; React app shell (~80 KB gzipped) + schools.json (60,421 schools, 16 fields, ~6 MB gzipped). ~6 TB total bandwidth during enrollment (1M unique learners × ~6 MB). Blaze plan. Storage <1 GB.',
   '1 enrollment', '₱55,800',
   '6,000 GB × $0.15/GB = $900 = ₱55,800. Cost is incurred almost entirely during the enrollment window — ~₱0 off-season. schools.json cached after first load; cost reflects 1M unique first-time visitors.',
   'Delivers the Student View to learners\' browsers. The dominant transfer cost is schools.json — a pre-built static file of all 60,421 Philippine schools (16 fields, ~6 MB gzipped) loaded once per learner. All school search and filtering runs client-side with no API calls.'],

  ['Firebase Hosting  (School Portal)',
   '₱0/month',
   'Global CDN; React app shell (~80 KB gzipped) only. ~500 school admins × 80 KB = ~40 MB total bandwidth for the year. Well within 360 MB/day free tier.',
   '12 months', '₱0',
   'School admin traffic is negligible — ~500 admins vs 1M learners. Free tier (360 MB/day) is never approached.',
   'Delivers the School Portal to school administrators\' browsers. Admins review incoming student applications, confirm or decline placements, and export enrollment data. No large static data file needed — all applicant data is fetched from the API per school.'],

  ['Firebase Authentication',
   '₱0/month',
   'Free tier: 50,000 MAU/month. Only school admins (~100–500 users) have Firebase Auth accounts. Students use stateless LRN validation — no Firebase Auth account created per learner.',
   '12 months', '₱0',
   'Budget assumes NO per-learner accounts. If added in future: ~₱324,000/month during enrollment (950K MAU × $0.0055).',
   'Secures the School Portal — only verified school administrators can view student applications and manage slot confirmations. Students are identified by LRN only; no learner login is required.'],

  ['Cloud Run  (Student View API, asia-southeast1)',
   '₱3,500/enrollment',
   '~3.5M total API requests during enrollment window: LRN validation (1M), draft saves (1.5M), final submissions (1M). min-instances: 1 for 90-day window; max-instances: 50. 1 vCPU (required — Cloud Run enforces max 1 concurrent request per instance for <1 vCPU); 256 MB RAM.',
   '1 enrollment', '₱3,500',
   'Rates: CPU active $0.0000336/vCPU-sec, CPU idle $0.0000035/vCPU-sec, Memory $0.0000035/GiB-sec, Requests $0.40/M (asia-southeast1, no free tier). Breakdown: CPU active 700K vCPU-sec × $0.0000336 = $23.52; CPU idle 7.08M vCPU-sec × $0.0000035 = $24.77; Memory active+idle = $6.80; Requests 3.5M × $0.40/M = $1.40. Total ~$56.50 = ₱3,502 → ₱3,500. Schools.json is served by Firebase Hosting — not Cloud Run. Scales to zero off-season.',
   'Handles every transactional action for learners: LRN validation against DepEd EBEIS, saving draft ranked choices, and final submission of applications. School browsing requires no API call — served from the static schools.json file loaded on app open.'],

  ['Cloud Run  (School Portal API, asia-southeast1)',
   '₱100/enrollment',
   '~100K total API requests during enrollment window: view applicants, confirm/decline, slot dashboard, CSV export. ~500 school admins × ~200 actions. 0.25 vCPU; 256 MB RAM. min-instances: 0 (cold starts acceptable for admin workflow).',
   '1 enrollment', '₱100',
   'Rates: CPU active $0.0000336/vCPU-sec (asia-southeast1). Breakdown: CPU active 100K × 0.2s × 0.25 vCPU × $0.0000336 = $0.17; Memory = $0.02; Requests 100K × $0.40/M = $0.04. Total ~$0.23 = ₱14. Rounded to ₱100 for contingency. School Portal admin traffic (500 users) is negligible vs. Student View (1M learners). Scales to zero off-season.',
   'Handles all School Portal actions: loading the ranked-choice applicant list per school, processing accept/decline/waitlist decisions, updating slot counts, and generating CSV exports of confirmed enrollees for GASS records.'],

  ['Compute Engine e2-standard-2  (OSRM road routing server)',
   '₱5,736/month',
   '2 vCPU, 8 GB RAM; asia-southeast1; $0.0767/hour on-demand. Runs for 90-day enrollment window only; shut down off-season. Hosts OSRM with Philippines OSM road network (~3 GB loaded in memory).',
   '3 months', '₱10,272',
   '$0.0767/hour × 2,160 hrs (90 days) = $165.67 = ₱10,272. Compute Engine required — Cloud Run cannot hold the 3 GB road network in memory persistently (cold starts take 30–60 sec loading the routing graph). Compute Engine source: cloud.google.com/compute/all-pricing.',
   'Self-hosted road routing engine. Computes road distance (km) and estimated travel time between a learner\'s location and each school they view in detail. Using OSRM (open-source, free) instead of Google Maps Distance Matrix API saves ~₱3,100,000 (Google Maps would cost $5 per 1,000 pairs × 10M pairs for 1M learners × 10 school views).'],

  ['Cloud SQL for PostgreSQL  (2 vCPU, 7.5 GB RAM)',
   '₱9,247/month',
   '2 vCPU, 7.5 GiB RAM; 50 GB SSD; asia-southeast1; always-on. Enterprise Edition — General Purpose. Rates: $0.0578/vCPU/hr + $0.0098/GiB/hr + $0.222/GB/month SSD.',
   '12 months', '₱111,000',
   'vCPU: 2 × $0.0578 × 730 = $84.39/mo. Memory: 7.5 GiB × $0.0098 × 730 = $53.66/mo. Storage: 50 GB × $0.222 = $11.10/mo. Total: $149.15/mo × 12 × ₱62 = ₱110,960 → ₱111,000. Source: cloud.google.com/sql/pricing [Source 3].',
   'The authoritative database for all pilot data: student ranked-choice submissions keyed by LRN, ESC slot allocations from GASS, school admin accounts, and enrollment audit logs. Persists data year-round across enrollment windows.'],

  ['Cloud SQL Automated Backups  (90-day retention)',
   '₱620/month',
   '90-day automated daily backup retention; PITR enabled; ~110 GB backup storage at steady state (90 compressed snapshots × ~1 GB + WAL logs); $0.08/GB/month.',
   '12 months', '₱7,440',
   'Cloud SQL built-in — no separate product. Extends default 7-day retention to 90 days via Cloud SQL settings.',
   'Protects student submission data against accidental deletion or system failure. Enables full recovery to any point within 90 days — covering the enrollment window plus a post-close period for dispute resolution, as required for government enrollment records.'],

  ['PgBouncer  (connection pooler, Cloud Run service)',
   '₱620/month',
   'Lightweight pooler as a permanent Cloud Run service; min-instances: 1. Routes Cloud Run instances through a shared pool of 20–30 DB connections.',
   '12 months', '₱7,440',
   'Required at 1M learner scale. Without it, 100+ Cloud Run instances × 5 connections = Cloud SQL exhaustion.',
   'Allows Cloud Run to auto-scale to 100+ instances without crashing the database. Acts as a traffic manager between the API and Cloud SQL — critical for handling the deadline-day enrollment burst without service interruption.'],

  ['Cloud Run Jobs  (ILP optimization engine)',
   '₱155/month',
   'Async batch compute for OR-Tools / PuLP optimization runs. 2 vCPU, 8 GB RAM per job; ~30–60 min per run. Invoked by GASS planning staff.',
   '12 months', '₱1,860',
   '~$0.12/run; estimated 10 runs/month max. Async — avoids Cloud Run\'s 60-min HTTP timeout.',
   'Runs GASS\'s ESC slot allocation optimization scenarios as background batch jobs, separate from the student-facing API. Ensures heavy computation (30–60 min) does not affect student enrollment availability or response times.'],

  ['Cloud Storage  (GeoJSON, school data files)',
   '₱310/month',
   'Standard storage; school coordinates, GeoJSON boundaries, ESC slot exports; <10 GB; $0.020/GB/month; asia-southeast1.',
   '12 months', '₱3,720',
   'Minimal footprint. Costs dominated by Cloud SQL and hosting.',
   'Stores the geographic and school reference data used across the system: school map coordinates, regional boundary files for the Student View map, and ESC slot allocation data exports for GASS.'],

  ['BigQuery  (pilot-scale analytics)',
   '₱155/month',
   'On-demand queries for GASS reporting; <1 TB queries/month at $6.25/TB. Storage <10 GB at $0.020/GB/month. Free tier: 10 GB storage, 1 TB queries/month.',
   '12 months', '₱1,860',
   'Likely within or near free tier at pilot scale.',
   'Provides GASS with aggregate pilot monitoring — total submissions, slot fill rates by school, geographic distribution of applications, and rank preference breakdowns — without querying the live production database and affecting enrollment performance.'],

  ['Egress, Cloud Monitoring, Secret Manager',
   '₱620/month',
   'Internet egress ($0.19/GB); Cloud Monitoring (free tier alerting + dashboards); Secret Manager for API keys ($0.06/secret/month).',
   '12 months', '₱7,440',
   'Egress driven by API responses to 1M learners during enrollment.',
   '(1) Egress — pays for data transmitted from GCP to learners\' devices during enrollment; (2) Cloud Monitoring — alerts E-CAIR and GASS if the system goes down during the enrollment window; (3) Secret Manager — securely stores DepEd EBEIS API credentials so they are never hardcoded in the application.'],
].forEach((row, i) => { dataRow(ws2, row, NC2, i % 2 === 0).height = 52; });

{
  const r = dataRow(ws2, ['GCP Subtotal', '', '', '', '₱210,332', '', ''], NC2, false, true, TOTALROW);
  r.height = 20;
  r.getCell(5).font = { bold: true, size: 11 };
}
{
  const r = dataRow(ws2, ['GCP Budget  (rounded, with ~20% contingency)', '', 'Contingency covers scope expansion, Firebase bandwidth overrun, or Cloud SQL tier upgrade.', '', '₱255,000', 'Recommended DBM line item: Cloud Compute and Storage Services (MOOE).', ''], NC2, false, true, DEPEDBLUE);
  r.height = 28;
  for (let c = 1; c <= NC2; c++) r.getCell(c).font = { bold: true, size: 10, color: { argb: TEXTWHITE } };
}
blank(ws2);

// ── SCENARIO B: 1M True Concurrent ────────────────────────────────────────────
subSectionTitle(ws2, '  SCENARIO B — Remove Concurrency Limit  (All 1M Learners May Access and Submit Simultaneously)', NC2);
{
  const note = ws2.addRow(['Scenario A (above) handles up to ~20,000–30,000 simultaneous submitting users and requires staggered access by DepEd division. Scenario B removes this constraint. Only changed line items are shown — all other components (Firebase Hosting, Firebase Auth, PgBouncer, Cloud Run Jobs, Cloud Storage, BigQuery, Egress) remain the same as Scenario A.']);
  merge(ws2, note.number, 1, note.number, NC2);
  note.height = 32;
  note.getCell(1).font      = { italic: true, size: 10 };
  note.getCell(1).alignment = { vertical: 'top', wrapText: true, indent: 1 };
}
colHeaders(ws2, ['Cloud Compute and Storage Services', 'Unit Cost', 'Specs', 'Qty', 'Amount', 'Notes', 'Purpose'], NC2);

[
  ['Compute Engine  (OSRM × 10 instances, load balanced)',
   '₱102,734/enrollment',
   '10 × e2-standard-2 (2 vCPU, 8 GB RAM each) behind a Cloud Load Balancer; asia-southeast1. Handles ~5,500 routing queries/second (1M users × 10 school views ÷ 1,800 sec). Runs 90-day enrollment window only.',
   '1 enrollment', '₱102,734',
   '10 × $0.0767/hour × 2,160 hours = $1,657 = ₱102,734. vs. Scenario A: ₱10,272 (1 instance). Additional cost: +₱92,462. Requires Cloud Load Balancer in front of instance group.',
   'Scales the road routing engine from 1 to 10 instances to handle peak concurrent distance queries from all 1M learners browsing simultaneously. Without this, a single OSRM instance saturates at ~1,000 queries/second — insufficient for 1M concurrent users.'],

  ['Cloud SQL for PostgreSQL  (32 vCPU, 120 GiB RAM — enrollment window)',
   '₱405,129/enrollment\n+ ₱85,317 off-season & storage',
   'db-custom-32-122880; asia-southeast1. Rates: $0.0578/vCPU/hr + $0.0098/GiB/hr [Source 3]. Scaled up for 90-day enrollment window only — downscaled to 2 vCPU off-season. Handles ~1,900 DB writes/second sustained (1M users × 3.5 API calls ÷ 1,800 sec).',
   '1 year', '₱490,000',
   'Enrollment 2,160 hrs: (32 × $0.0578 + 120 × $0.0098) × 2,160 = $3.026/hr × 2,160 = $6,534 = ₱405,129. Off-season 9 months (2 vCPU/7.5 GiB): $1,243 = ₱77,066. Storage $0.222/GB × 50 GB × 12 = $133 = ₱8,258. Total: $7,910 × ₱62 = ₱490,446 → ₱490,000. vs. Scenario A: ₱111,000. Additional cost: +₱379,000. Scale-up/down requires planned maintenance window.',
   'Upgrades the database to handle ~1,900 concurrent writes/second — the throughput required when all 1M learners are submitting simultaneously. The 2 vCPU instance in Scenario A caps at ~200–400 writes/second; this 32 vCPU instance removes that bottleneck entirely.'],
].forEach((row, i) => { dataRow(ws2, row, NC2, i % 2 === 0).height = 60; });

{
  const r = dataRow(ws2, ['GCP Subtotal — Scenario B', '', '(Scenario A unchanged items: ₱210,332 − ₱10,272 OSRM − ₱111,000 Cloud SQL) + ₱102,734 OSRM × 10 + ₱490,000 Cloud SQL 32 vCPU', '', '₱681,794', '', ''], NC2, false, true, TOTALROW);
  r.height = 28;
  r.getCell(5).font = { bold: true, size: 11 };
}
{
  const r = dataRow(ws2, ['GCP Budget — Scenario B  (rounded, with ~20% contingency)', '', 'Same contingency buffer as Scenario A. Covers Cloud Load Balancer setup for OSRM and Cloud SQL scale-up/down operational overhead.', '', '₱825,000', 'vs. Scenario A: ₱255,000. Additional GCP cost for removing concurrency limit: +₱570,000.', ''], NC2, false, true, DEPEDBLUE);
  r.height = 28;
  for (let c = 1; c <= NC2; c++) r.getCell(c).font = { bold: true, size: 10, color: { argb: TEXTWHITE } };
}
{
  const r = dataRow(ws2, [
    '★  SCENARIO B — RECOMMENDED FY 2027 BUDGET ASK  (MOOE)', '',
    'Removes the 20,000–30,000 simultaneous user limit. No staggered scheduling required. All 1M learners may access and submit at any time during the enrollment window.',
    '', '₱3,100,000',
    'vs. Scenario A: ₱2,500,000. Additional cost: +₱570,000 GCP (Cloud SQL 32 vCPU + OSRM × 10 instances). E-CAIR and Operations components unchanged.',
    'Funds the full pilot with no operational constraints on concurrent access — any learner can submit at any time without risk of system degradation.',
  ], NC2, false, true, RECROW);
  r.height = 44;
  for (let c = 1; c <= NC2; c++) r.getCell(c).font = { bold: true, size: 11, color: { argb: '7F6000' } };
}

blank(ws2); blank(ws2);

// B. E-CAIR Professional Services
subSectionTitle(ws2, '  B. Professional Services — E-CAIR Maintenance and Support Retainer  (12-month)', NC2);
{
  const note = ws2.addRow(['Scope: break-fix, security patches, minor enhancements, DevOps/monitoring, GASS technical support. ~30–40 hours/month. Subject to E-CAIR\'s formal proposal.']);
  merge(ws2, note.number, 1, note.number, NC2);
  note.height = 18;
  note.getCell(1).font      = { italic: true, size: 9, color: { argb: TEXTGREY } };
  note.getCell(1).alignment = { vertical: 'top', wrapText: true, indent: 1 };
}
colHeaders(ws2, ['Service', 'Unit Cost', 'Specs', 'Qty', 'Amount', 'Notes', 'Purpose'], NC2);
{
  const r = dataRow(ws2, [
    'Maintenance and support retainer (E-CAIR)',
    '₱60,000–120,000/month',
    'Bug fixes; security patches; minor feature enhancements; DevOps monitoring and incident response; GASS technical support calls.',
    '12 months', '₱720,000–1,440,000',
    'Rate confirmed via E-CAIR\'s formal proposal. Range based on Philippine IT consulting rates.',
    'Ensures the platform remains operational, secure, and responsive to GASS feedback throughout the pilot year. Provides the technical expertise required to maintain a government-grade enrollment system that 1M learners depend on — covering both routine upkeep and incident response.',
  ], NC2, true);
  r.height = 52;
}
blank(ws2);

// C. GASS Operations
subSectionTitle(ws2, '  C. GASS Operations', NC2);
colHeaders(ws2, ['Item', 'Unit Cost', 'Specs', 'Qty', 'Amount', 'Notes', 'Purpose'], NC2);

[
  ['Helpdesk support  (contracted, part-time)',
   '₱15,000–25,000/month',
   '1 staff handling school admin and student inquiries during pilot enrollment window. Agency or BPO contract.',
   '12 months', '₱180,000–300,000',
   'Handles Tier 1 issues. E-CAIR escalation for technical issues.',
   'Ensures school administrators and learners receive timely support during the enrollment window. Without dedicated helpdesk coverage, routine questions (login errors, submission issues, slot queries) would escalate to E-CAIR and delay resolution.'],

  ['School admin training and onboarding',
   '₱80,000–150,000 total',
   'Live sessions (in-person or virtual), printed quick-start guides, recorded walkthrough videos, Q&A documentation.',
   '1 (one-time)', '₱80,000–150,000',
   'One-time for pilot schools. Will need to scale for national rollout.',
   'Enables school staff to independently operate the School Portal — reviewing student applications, confirming placements, managing slots, and exporting enrollment data. A trained school admin base is a prerequisite to pilot launch; without it, schools cannot participate.'],

  ['Contingency (10% of total budget)',
   '—', '10% buffer for unexpected scope, additional training rounds, or vendor cost changes.', '—', '₱118,000–219,000',
   '10% applied to full budget total (GCP + E-CAIR + Operations).',
   'Provides budget flexibility to absorb unexpected costs without requiring a supplemental appropriation — e.g., scope expansion to more schools, additional training rounds, or infrastructure tier upgrades triggered by higher-than-expected traffic.'],
].forEach((row, i) => { dataRow(ws2, row, NC2, i % 2 === 0).height = 52; });

{
  const r = dataRow(ws2, ['Operations Subtotal', '', '', '', '₱378,000–669,000', '', ''], NC2, false, true, TOTALROW);
  r.height = 18;
  r.getCell(5).font = { bold: true, size: 10 };
}
blank(ws2);

// 2027 Grand Total
subSectionTitle(ws2, '  2027 GRAND TOTAL', NC2);
colHeaders(ws2, ['Category', 'Unit Cost', 'Description', 'Qty', 'Amount', 'Notes', 'Purpose'], NC2);

[
  ['A. Cloud Compute and Storage Services', '₱255,000/year', 'Firebase Hosting (schools.json + app shell), Cloud Run (enrollment API), Compute Engine (OSRM routing), Cloud SQL + Backups, PgBouncer, Cloud Run Jobs, Storage, BigQuery, Egress, Monitoring.', '1 year', '₱255,000', 'Hard cost; backed by Google official pricing. See Sources. Includes Compute Engine for OSRM road routing server (90-day enrollment window).', 'Provides the cloud infrastructure that runs the Student View, School Portal, and optimization engine — the technical foundation of the entire pilot.'],
  ['B. E-CAIR Professional Services', '₱60k–120k/month', '12-month maintenance and support retainer.', '12 months', '₱720,000–1,440,000', 'Subject to E-CAIR\'s proposed rate.', 'Keeps the system operational and evolving throughout the pilot year, with E-CAIR responsible for technical uptime and GASS able to focus on policy operations.'],
  ['C. GASS Operations', '—', 'Helpdesk, school admin training, contingency.', '—', '₱378,000–669,000', '', 'Ensures schools and learners can participate effectively in the pilot — trained admins, supported users, and budget flexibility for the unexpected.'],
  ['TOTAL', '', '', '', '₱1,353,000–2,364,000', '', ''],
].forEach((row, i) => {
  const isTotal = row[0] === 'TOTAL';
  const r = dataRow(ws2, row, NC2, i % 2 === 0, isTotal, isTotal ? TOTALROW : null);
  r.height = 28;
  if (isTotal) r.getCell(5).font = { bold: true, size: 11 };
});

{
  const r = dataRow(ws2, [
    '★  SCENARIO A — RECOMMENDED FY 2027 BUDGET ASK  (MOOE)', '',
    'Staggered enrollment by DepEd division. Max ~20,000–30,000 simultaneous submitting users. Accommodates upper bound of E-CAIR rates, scope expansion, and training contingency.',
    '', '₱2,500,000',
    'GCP: ₱255,000 | E-CAIR: up to ₱1,440,000 | Operations: up to ₱669,000 | Contingency buffer included.',
    'Funds the full production and pilot deployment of PAARAL with staggered access scheduling — enabling DepEd to run a credible, data-backed ESC enrollment pilot for 1 million learners.',
  ], NC2, false, true, RECROW);
  r.height = 44;
  for (let c = 1; c <= NC2; c++) r.getCell(c).font = { bold: true, size: 11, color: { argb: '7F6000' } };
  const r2 = dataRow(ws2, [
    '★  SCENARIO B — RECOMMENDED FY 2027 BUDGET ASK  (MOOE)', '',
    'No staggering required. All 1M learners may access and submit simultaneously. OSRM scaled to 10 instances; Cloud SQL upgraded to 32 vCPU during enrollment window.',
    '', '₱2,875,000',
    'GCP: ₱825,000 | E-CAIR: up to ₱1,440,000 | Operations: up to ₱669,000 | Additional vs. Scenario A: +₱570,000 GCP.',
    'Funds the full pilot with no operational constraints on concurrent access — eliminates the need for GASS to coordinate staggered enrollment windows by division.',
  ], NC2, false, true, RECROW);
  r2.height = 44;
  for (let c = 1; c <= NC2; c++) r2.getCell(c).font = { bold: true, size: 11, color: { argb: '7F6000' } };
}

blank(ws2); blank(ws2);

// ── 2028+: NATIONAL ROLLOUT (INDICATIVE) ──────────────────────────────────────
sectionTitle(ws2, '  2028+ — NATIONAL ROLLOUT  (Indicative; for Sustainability Plan)', NC2);
{
  const note = ws2.addRow(['Final amounts depend on national scope (10,000+ schools, 500,000+ learners). To be refined in the sustainability plan after pilot data is available.']);
  merge(ws2, note.number, 1, note.number, NC2);
  note.height = 16;
  note.getCell(1).font      = { italic: true, size: 10 };
  note.getCell(1).alignment = { vertical: 'middle', indent: 1 };
}
blank(ws2);

subSectionTitle(ws2, '  Cloud Compute and Storage Services — National Scale', NC2);
colHeaders(ws2, ['Cloud Compute and Storage Services', 'Unit Cost', 'Specs', 'Qty', 'Amount', 'Notes', 'Purpose'], NC2);

[
  ['Firebase Hosting  (national traffic)', '₱3,100–6,200/month', 'Exceeds free transfer tier; ~10 GB+/day bandwidth at national scale.', '12 months', '₱37,200–74,400', 'CDN-cached but first-load bandwidth scales with unique user count.', 'Delivers the Student View to all ESC-eligible Grade 6 learners nationwide. CDN distributes the app globally so even learners in Mindanao and Visayas get fast load times.'],
  ['Cloud Run  (national enrollment traffic)', '₱3,100–9,920/month', 'Millions of req/month during June–July enrollment peak; higher min-instances.', '12 months', '₱37,200–119,040', 'Auto-scales with traffic; no VM pre-provisioning.', 'Handles enrollment requests from all participating schools and learners across all DepEd divisions. Auto-scaling absorbs the June–July Grade 6 enrollment peak without pre-provisioning capacity.'],
  ['Cloud SQL 16+ vCPU / 60 GB RAM  or  Cloud Spanner', '₱24,800–49,600/month', 'Cloud SQL 16 vCPU / 60 GB RAM ($400–600/month) handles ~1,000 concurrent connections. Cloud Spanner (1 node, $648/month) provides unlimited horizontal scaling. Final choice determined by pilot concurrency data.', '12 months', '₱297,600–595,200', 'Upgrade required for national-scale concurrent load. 2027 pilot infrastructure (2 vCPU) caps at ~20,000–30,000 simultaneous submitting users — insufficient for national rollout without staggering. Cloud Spanner removes the concurrency ceiling entirely.', 'Handles concurrent writes from all DepEd regions simultaneously during the national enrollment peak. The 2027 Cloud SQL instance is the binding concurrency constraint for the pilot — this upgrade removes it for national rollout, allowing all learners to submit without staggered scheduling.'],
  ['BigQuery  (Planning View analytics)', '₱6,200–24,800/month', 'Heavy query volume for DepEd Planning View optimization simulations on full national dataset.', '12 months', '₱74,400–297,600', 'On-demand: $6.25/TB. Cost depends on GASS query frequency.', 'Powers the DepEd Planning View — enabling GASS to run ILP optimization simulations on the full national student flow dataset (29,000+ school-pair flows) for evidence-based ESC slot allocation decisions.'],
  ['Cloud Storage + CDN', '₱620–1,240/month', 'School registry, national GeoJSON, ESC slot history; CDN for faster regional delivery.', '12 months', '₱7,440–14,880', 'Storage grows slowly; CDN reduces Cloud Run egress costs.', 'Hosts the complete national school registry and geographic data, and distributes it efficiently to regional offices and school division users via CDN caching.'],
  ['Egress + Monitoring + PgBouncer', '₱1,550–3,720/month', 'Higher internet egress at national scale; enhanced monitoring; connection pooling for national load.', '12 months', '₱18,600–44,640', 'Monitoring includes alerting across all DepEd divisions.', 'Sustains national-scale operations: data transfer to all regions, system health monitoring across all DepEd divisions, and database connection management at national concurrent load.'],
].forEach((row, i) => { dataRow(ws2, row, NC2, i % 2 === 0).height = 44; });

{
  const r = dataRow(ws2, ['GCP Subtotal', '', '', '', '₱472,440–1,145,760', '', ''], NC2, false, true, TOTALROW);
  r.height = 18;
  r.getCell(5).font = { bold: true, size: 10 };
}
blank(ws2);

subSectionTitle(ws2, '  Full Annual Budget — Indicative (2028)', NC2);
colHeaders(ws2, ['Category', 'Unit Cost', 'Description', 'Qty', 'Amount', 'Notes', 'Purpose'], NC2);

[
  ['GCP Infrastructure', '—', 'See GCP table above. Includes Cloud SQL 16+ vCPU or Cloud Spanner for national-scale concurrency.', '1 year', '₱473,000–1,146,000', 'Hard infrastructure cost. Cloud SQL/Spanner upgrade is the largest cost driver vs. 2027 pilot.', 'Provides the cloud foundation for national-scale enrollment, school administration, and GASS analytics across all DepEd regions. Concurrency constraint from 2027 pilot is removed by the database upgrade.'],
  ['DepEd Planning View development  (one-time)', '—', 'BigQuery pipeline + real ILP optimizer (OR-Tools or Gurobi). One-time.', '1', '₱1,000,000–2,000,000', 'May be classified as Capital Outlay (CO) per DBM guidelines.', 'Builds the production DepEd Planning View with real national data and a mathematical optimizer — enabling GASS to make evidence-based ESC slot allocation decisions rather than relying on estimates or historical patterns.'],
  ['Platform maintenance contract', '—', 'Larger scope: national school count, full data pipeline, all three views.', '1 year', '₱1,440,000–3,600,000', 'E-CAIR long-term contract or DepEd in-house team.', 'Sustains all three PAARAL views (Student, School, DepEd Planning) at national scale and incorporates feedback from the 2027 pilot — ensuring the system continues to improve after turnover.'],
  ['Helpdesk + school support operations', '—', '2–3 full-time staff or contracted BPO, scaled to national school count.', '1 year', '₱500,000–1,000,000', '', 'Provides support coverage for all participating schools and students across all DepEd divisions — ensuring no school is left without assistance during the national enrollment window.'],
  ['Training + change management  (national)', '—', 'Per-division rollout training; printed materials, online guides, live sessions.', '1 year', '₱200,000–500,000', 'All DepEd divisions; ~200+ divisions nationwide.', 'Ensures all DepEd divisions and school administrators can independently use PAARAL without relying on E-CAIR for routine operations — a prerequisite for sustainable national deployment.'],
  ['2028 ANNUAL TOTAL  (indicative)', '', '', '', '₱3,615,000–8,248,000', 'To be refined in sustainability plan once pilot data is available. Increase from 2027 driven primarily by database upgrade for national-scale concurrency.', ''],
].forEach((row, i) => {
  const isTotal = row[0].includes('TOTAL');
  const r = dataRow(ws2, row, NC2, i % 2 === 0, isTotal, isTotal ? TOTALROW : null);
  r.height = 44;
  if (isTotal) r.getCell(5).font = { bold: true, size: 11 };
});

blank(ws2); blank(ws2);

// ── SOURCES ────────────────────────────────────────────────────────────────────
sectionTitle(ws2, '  SOURCES', NC2);
{
  const note = ws2.addRow(['All infrastructure cost estimates are derived from official Google Cloud pricing pages and verified third-party sources. Exchange rate from BSP reference.']);
  merge(ws2, note.number, 1, note.number, NC2);
  note.height = 16;
  note.getCell(1).font      = { italic: true, size: 10 };
  note.getCell(1).alignment = { vertical: 'middle', indent: 1 };
}
colHeaders(ws2, ['#', 'Source', 'URL', '', 'Used to Price', '', ''], NC2);
merge(ws2, ws2.lastRow.number, 3, ws2.lastRow.number, 4);
merge(ws2, ws2.lastRow.number, 5, ws2.lastRow.number, NC2);

[
  { num: '1', name: 'Google Cloud Run Pricing',                       url: 'https://cloud.google.com/run/pricing',                                              note: 'Cloud Run asia-southeast1 (Tier 2) request-based billing: CPU active $0.0000336/vCPU-second, CPU idle (min-instance) $0.0000035/vCPU-second, Memory active/idle $0.0000035/GiB-second, Requests $0.40/million. Free tier (2M req/month, 180K vCPU-sec, 360K GiB-sec) applies in US regions only — NOT in asia-southeast1.' },
  { num: '2', name: 'Google Cloud SQL Pricing — Instance Rates',        url: 'https://cloud.google.com/sql/pricing',                                              note: 'Enterprise Edition, General Purpose, asia-southeast1 (on-demand): vCPU $0.0578/vCPU-hour, Memory $0.0098/GiB-hour. CUD 1-year: $0.04335/vCPU-hour, $0.00735/GiB-hour. HA rates are 2× standard. Verified May 2026 from official pricing page.' },
  { num: '3', name: 'Google Cloud SQL Pricing — Storage & Egress',      url: 'https://cloud.google.com/sql/pricing',                                              note: 'SSD storage: $0.222/GB/month. Backup storage: $0.08/GB/month. Egress: $0.19/GB to internet. No free tier for dedicated instances.' },
  { num: '4', name: 'Firebase Pricing',                                url: 'https://firebase.google.com/pricing',                                               note: 'Hosting free tier: 10 GB storage, 360 MB/day transfer; overage $0.026/GB storage, $0.15/GB transfer. Auth free tier: 50,000 MAU/month; beyond: $0.0055/MAU via Cloud Identity Platform.' },
  { num: '5', name: 'Google BigQuery Pricing',                         url: 'https://cloud.google.com/bigquery/pricing',                                         note: 'On-demand queries: $6.25/TB scanned. Storage: $0.020/GB/month active, $0.010/GB/month long-term. Free tier: 1 TB/month queries, 10 GB/month storage.' },
  { num: '6', name: 'BSP / PhilNews — USD/PHP Rate, May 1, 2026',      url: 'https://philnews.ph/2026/05/01/usd-to-php-exchange-rate-today-friday-may-1-2026', note: 'BSP reference rate: ₱61.42/USD on May 1, 2026. Budget uses ₱62/USD as conservative planning rate for 2027.' },
  { num: '7', name: 'Google Compute Engine Pricing',                    url: 'https://cloud.google.com/compute/all-pricing',                                       note: 'e2-standard-2 (2 vCPU, 8 GB RAM) on-demand price in asia-southeast1: $0.0767/hour. Used to price the OSRM road routing server (90-day enrollment window = 2,160 hours = $165.67 = ₱10,272).' },
].forEach((s, i) => {
  const row = ws2.addRow([s.num, s.name, s.url, '', s.note, '', '']);
  merge(ws2, row.number, 3, row.number, 4);
  merge(ws2, row.number, 5, row.number, NC2);
  row.height = 36;
  applyRowStyle(row, NC2, i % 2 === 0);
  row.getCell(1).font = { bold: true, size: 10 };
  const urlCell   = row.getCell(3);
  urlCell.value   = { text: s.url, hyperlink: s.url };
  urlCell.font    = { size: 10, color: { argb: LINKBLUE }, underline: true };
});

blank(ws2);
{
  const r = ws2.addRow(['Document prepared by E-CAIR  |  Contact: paumartinez.work@gmail.com  |  Prices subject to change; verify on official pricing pages before DBM submission.']);
  merge(ws2, r.number, 1, r.number, NC2);
  r.height = 16;
  r.getCell(1).font      = { italic: true, size: 9, color: { argb: TEXTGREY } };
  r.getCell(1).alignment = { vertical: 'middle', indent: 1 };
}

// ════════════════════════════════════════════════════════════════════════════
// SHEET 3: CALCULATIONS
// Detailed arithmetic for each Budget tab line item.
// ════════════════════════════════════════════════════════════════════════════
const ws3 = wb.addWorksheet('Calculations', { views: [{ showGridLines: false }] });
const NC3 = 5;
ws3.columns = [
  { key: 'a', width: 38 },  // Component / Parameter
  { key: 'b', width: 24 },  // Rate
  { key: 'c', width: 26 },  // × Quantity
  { key: 'd', width: 14 },  // = USD
  { key: 'e', width: 40 },  // Notes / Source
];

{
  const r = ws3.addRow(['PAARAL — Cost Calculation Workbook']);
  r.height = 32;
  merge(ws3, r.number, 1, r.number, NC3);
  r.getCell(1).font      = { bold: true, size: 16, color: { argb: NAVY } };
  r.getCell(1).alignment = { vertical: 'middle' };
}
{
  const r = ws3.addRow(['Detailed arithmetic behind each Budget tab line item. Exchange rate: ₱62/USD (BSP reference May 1, 2026: ₱61.42/USD). GCP region: asia-southeast1.']);
  r.height = 16;
  merge(ws3, r.number, 1, r.number, NC3);
  r.getCell(1).font = { italic: true, size: 9, color: { argb: TEXTGREY } };
}
blank(ws3);

// ── Calculation helpers ───────────────────────────────────────────────────────
const INPUTBG  = 'FFF5F5F5';
const INPUTBG2 = 'FFE8E8E8';

function calcSubHeader(ws, text) {
  const row = ws.addRow([text]);
  row.height = 15;
  merge(ws, row.number, 1, row.number, NC3);
  applyRowStyle(row, NC3, false, false, LIGHTBLUE);
  row.getCell(1).font      = { bold: true, size: 9, color: { argb: TEXTDARK } };
  row.getCell(1).alignment = { vertical: 'middle', horizontal: 'left', indent: 2 };
}

function calcColHeaders(ws) {
  return colHeaders(ws, ['Component / Parameter', 'Rate', '× Quantity', '= USD', 'Notes / Source'], NC3);
}

function inputRow(ws, param, value, note, alt) {
  const row = ws.addRow([param, value, '', '', note]);
  row.height = 20;
  merge(ws, row.number, 3, row.number, 4);
  applyRowStyle(row, NC3, false, false, alt ? INPUTBG2 : INPUTBG);
  row.getCell(1).font = { size: 9, color: { argb: TEXTGREY } };
  row.getCell(2).font = { bold: true, size: 10 };
  row.getCell(5).font = { italic: true, size: 9, color: { argb: TEXTGREY } };
  return row;
}

function calcRow(ws, component, rate, qty, usd, note, alt) {
  const row = ws.addRow([component, rate, qty, usd, note]);
  row.height = 22;
  applyRowStyle(row, NC3, alt);
  row.getCell(4).font = { size: 10 };
  return row;
}

function usdSubtotal(ws, usd) {
  const row = ws.addRow(['  Subtotal (USD)', '', '', usd, '']);
  row.height = 18;
  merge(ws, row.number, 1, row.number, 3);
  applyRowStyle(row, NC3, false, true, TOTALROW);
  row.getCell(1).font = { bold: true, size: 10 };
  row.getCell(4).font = { bold: true, size: 10 };
}

function phpTotalRow(ws, usdAmt, phpAmt, budgetLabel) {
  const row = ws.addRow(['  × ₱62 / USD  =', usdAmt, phpAmt, '', budgetLabel]);
  row.height = 22;
  merge(ws, row.number, 3, row.number, 4);
  applyRowStyle(row, NC3, false, true, RECROW);
  for (let c = 1; c <= NC3; c++) row.getCell(c).font = { bold: true, size: 10, color: { argb: '7F6000' } };
  row.getCell(3).font = { bold: true, size: 11, color: { argb: '7F6000' } };
}

// ── 1. Firebase Hosting — Student View ───────────────────────────────────────
sectionTitle(ws3, '  1. Firebase Hosting — Student View  →  Budget tab: ₱55,800 / enrollment', NC3);
calcSubHeader(ws3, '  Inputs & Assumptions');
inputRow(ws3, 'Unique learners (pilot scope)', '1,000,000', 'All ESC-eligible Grade 6 learners. Confirmed jointly with GASS in Phase 0.', false);
inputRow(ws3, 'schools.json size (gzipped)', '6 MB', 'Conservative estimate. 60,421 schools × 16 static fields; ~30 MB uncompressed. 6 MB is the conservative gzipped figure.', true);
inputRow(ws3, 'App shell (gzipped)', '~80 KB', 'React bundle (JS + CSS). Negligible vs. schools.json; omitted from calculation.', false);
inputRow(ws3, 'Firebase Blaze plan — transfer overage rate', '$0.15 / GB', 'Source: firebase.google.com/pricing [Source 4]', true);
inputRow(ws3, 'Firebase free tier', '360 MB / day', 'Exhausted in minutes on enrollment day (1M learners × 6 MB = 6,000 GB >> 360 MB/day). All bandwidth billed at overage.', false);

calcSubHeader(ws3, '  Calculation');
calcColHeaders(ws3);
let a3 = false;
calcRow(ws3, 'Total bandwidth transferred', '$0.15 / GB', '6,000 GB', '$900.00', '1,000,000 learners × 6 MB ÷ 1,000 = 6,000 GB × $0.15/GB', a3);
usdSubtotal(ws3, '$900.00');
phpTotalRow(ws3, '$900.00', '₱55,800', '→ Budget tab: "Firebase Hosting (Student View)" = ₱55,800 / enrollment');
blank(ws3); blank(ws3);

// ── 2. Firebase Hosting — School Portal ──────────────────────────────────────
sectionTitle(ws3, '  2. Firebase Hosting — School Portal  →  Budget tab: ₱0', NC3);
calcSubHeader(ws3, '  Inputs & Assumptions');
inputRow(ws3, 'School admins (MAU)', '~500', 'One admin per participating school (~100–500 pilot schools).', false);
inputRow(ws3, 'App shell per load (gzipped)', '80 KB', 'React bundle. No large static file — applicant data fetched from the API per school.', true);
inputRow(ws3, 'Firebase free tier', '360 MB / day  (10 GB / month)', 'Applies globally to Firebase Hosting.', false);

calcSubHeader(ws3, '  Calculation');
calcColHeaders(ws3);
a3 = false;
calcRow(ws3, 'Total bandwidth transferred', '$0.15 / GB', '0.04 GB', '$0.00', '500 admins × 80 KB = 40 MB. Free tier = 360 MB/day >> 40 MB total for the year. No billing triggered.', a3);
usdSubtotal(ws3, '$0.00');
phpTotalRow(ws3, '$0.00', '₱0', '→ Budget tab: "Firebase Hosting (School Portal)" = ₱0');
blank(ws3); blank(ws3);

// ── 3. Firebase Authentication ────────────────────────────────────────────────
sectionTitle(ws3, '  3. Firebase Authentication  →  Budget tab: ₱0', NC3);
calcSubHeader(ws3, '  Inputs & Assumptions');
inputRow(ws3, 'School admin accounts (MAU)', '~500', 'Only school admins have Firebase Auth accounts. Authentication decision: students use stateless LRN validation — no Firebase Auth account created per learner.', false);
inputRow(ws3, 'Firebase Auth free tier', '50,000 MAU / month', 'Source: firebase.google.com/pricing [Source 4]', true);
inputRow(ws3, 'Cost if per-learner accounts were added', '~₱324,000 / month during enrollment', 'For reference only: 950K MAU × $0.0055/MAU = $5,225/month = ₱324,000/month. Architecture decision to use LRN-only avoids this cost.', false);

calcSubHeader(ws3, '  Calculation');
calcColHeaders(ws3);
a3 = false;
calcRow(ws3, 'School admin MAU vs. free tier', '$0.0055 / MAU (beyond 50K)', '~500 MAU', '$0.00', '500 MAU << 50,000 MAU free tier. No billing triggered.', a3);
usdSubtotal(ws3, '$0.00');
phpTotalRow(ws3, '$0.00', '₱0', '→ Budget tab: "Firebase Authentication" = ₱0');
blank(ws3); blank(ws3);

// ── 4. Cloud Run — Student View API ──────────────────────────────────────────
sectionTitle(ws3, '  4. Cloud Run — Student View API (asia-southeast1, request-based)  →  Budget tab: ₱3,500 / enrollment', NC3);
calcSubHeader(ws3, '  Inputs & Assumptions');
inputRow(ws3, 'Total API requests (enrollment window)', '3,500,000', 'LRN validation (1M) + draft saves (1.5M, avg 1.5 saves per learner) + final submissions (1M). School browsing served from static schools.json — no API call.', false);
inputRow(ws3, 'Average request duration', '0.2 seconds', 'Estimate: DB query via PgBouncer + LRN validation + JSON response. Latency dominated by Cloud SQL round-trip (~100–200 ms).', true);
inputRow(ws3, 'vCPU allocation', '1 vCPU', 'Required: Cloud Run enforces max 1 concurrent request per instance for <1 vCPU. 1 vCPU allows concurrency=80 (multiple requests handled per instance simultaneously).', false);
inputRow(ws3, 'Memory allocation', '256 MB = 0.25 GiB', 'Sufficient for a stateless Node.js/Python API handling LRN validation and DB queries.', true);
inputRow(ws3, 'Min-instances', '1 (for 90-day enrollment window)', 'Keeps 1 warm instance active throughout enrollment. Eliminates cold start latency for learners. Set to 0 off-season.', false);
inputRow(ws3, 'Enrollment window', '90 days = 7,776,000 seconds', 'January – March 2027. The min-instance idle cost accumulates for this entire period.', true);

calcSubHeader(ws3, '  Derived Values');
inputRow(ws3, 'Total active vCPU-seconds', '700,000 vCPU-sec', '3,500,000 requests × 0.2 sec × 1 vCPU', false);
inputRow(ws3, 'Total idle seconds (1 min-instance)', '7,076,000 sec', '7,776,000 total − 700,000 active = 7,076,000 idle', true);
inputRow(ws3, 'Total active memory GiB-seconds', '175,000 GiB-sec', '3,500,000 requests × 0.2 sec × 0.25 GiB', false);
inputRow(ws3, 'Total idle memory GiB-seconds (min-instance)', '1,769,000 GiB-sec', '7,076,000 idle sec × 0.25 GiB', true);

calcSubHeader(ws3, '  Cost Components  [Source 1: cloud.google.com/run/pricing — asia-southeast1 Tier 2]');
calcColHeaders(ws3);
a3 = false;
calcRow(ws3, 'Requests', '$0.40 / million', '3,500,000 req', '$1.40',  'No free tier in asia-southeast1 (free tier is US regions only).', a3); a3 = !a3;
calcRow(ws3, 'CPU — active time', '$0.0000336 / vCPU-sec', '700,000 vCPU-sec', '$23.52', '3.5M × 0.2s × 1 vCPU × $0.0000336', a3); a3 = !a3;
calcRow(ws3, 'CPU — idle  (1 min-instance, 90 days)', '$0.0000035 / vCPU-sec', '7,076,000 vCPU-sec', '$24.77', '(7,776,000 − 700,000) × 1 vCPU × $0.0000035   ← dominant cost: instance runs 24/7 during enrollment', a3); a3 = !a3;
calcRow(ws3, 'Memory — active time', '$0.0000035 / GiB-sec', '175,000 GiB-sec', '$0.61',  '3.5M × 0.2s × 0.25 GiB × $0.0000035', a3); a3 = !a3;
calcRow(ws3, 'Memory — idle  (1 min-instance, 90 days)', '$0.0000035 / GiB-sec', '1,769,000 GiB-sec', '$6.19',  '7,076,000 × 0.25 GiB × $0.0000035', a3);
usdSubtotal(ws3, '$56.49');
phpTotalRow(ws3, '$56.49', '₱3,502 → rounded ₱3,500', '→ Budget tab: "Cloud Run (Student View API)" = ₱3,500 / enrollment');
blank(ws3); blank(ws3);

// ── 5. Cloud Run — School Portal API ─────────────────────────────────────────
sectionTitle(ws3, '  5. Cloud Run — School Portal API (asia-southeast1, request-based)  →  Budget tab: ₱100 / enrollment', NC3);
calcSubHeader(ws3, '  Inputs & Assumptions');
inputRow(ws3, 'Total API requests (enrollment window)', '100,000', '~500 school admins × ~200 actions each: view applicant list, confirm/decline/waitlist, check slot dashboard, export CSV.', false);
inputRow(ws3, 'Average request duration', '0.2 seconds', 'Same estimate as Student View API.', true);
inputRow(ws3, 'vCPU allocation', '0.25 vCPU', 'Low-traffic service. Concurrency=1 per instance is acceptable — admins are not time-pressured the way students are.', false);
inputRow(ws3, 'Memory allocation', '256 MB = 0.25 GiB', '', true);
inputRow(ws3, 'Min-instances', '0', 'Cold starts (1–3 sec) are acceptable for school admins. No min-instance idle cost.', false);

calcSubHeader(ws3, '  Derived Values');
inputRow(ws3, 'Total active vCPU-seconds', '5,000 vCPU-sec', '100,000 × 0.2 sec × 0.25 vCPU', false);
inputRow(ws3, 'Total active memory GiB-seconds', '5,000 GiB-sec', '100,000 × 0.2 sec × 0.25 GiB', true);

calcSubHeader(ws3, '  Cost Components  [Source 1: cloud.google.com/run/pricing — asia-southeast1 Tier 2]');
calcColHeaders(ws3);
a3 = false;
calcRow(ws3, 'Requests', '$0.40 / million', '100,000 req', '$0.04',  'No free tier in asia-southeast1.', a3); a3 = !a3;
calcRow(ws3, 'CPU — active time', '$0.0000336 / vCPU-sec', '5,000 vCPU-sec', '$0.17',  '100K × 0.2s × 0.25 vCPU × $0.0000336', a3); a3 = !a3;
calcRow(ws3, 'Memory — active time', '$0.0000035 / GiB-sec', '5,000 GiB-sec', '$0.02',  '100K × 0.2s × 0.25 GiB × $0.0000035', a3);
usdSubtotal(ws3, '$0.23');
phpTotalRow(ws3, '$0.23', '₱14 → rounded ₱100', '→ Budget tab: "Cloud Run (School Portal API)" = ₱100 / enrollment. Rounded up for contingency.');
blank(ws3); blank(ws3);

// ── 6. Compute Engine — OSRM ─────────────────────────────────────────────────
sectionTitle(ws3, '  6. Compute Engine e2-standard-2 — OSRM Road Routing  →  Budget tab: ₱10,272 / enrollment', NC3);
calcSubHeader(ws3, '  Inputs & Assumptions');
inputRow(ws3, 'Instance type', 'e2-standard-2  (2 vCPU, 8 GB RAM)', 'Smallest general-purpose instance with sufficient RAM to hold the Philippines OpenStreetMap road network (~3 GB) in memory without swapping.', false);
inputRow(ws3, 'On-demand rate (asia-southeast1)', '$0.0767 / hour', 'Source: cloud.google.com/compute/all-pricing [Source 7]', true);
inputRow(ws3, 'Runtime', '90 days = 2,160 hours (enrollment window only)', 'Shut down off-season. No cost outside enrollment period.', false);
inputRow(ws3, 'Why Compute Engine — not Cloud Run', '3 GB routing graph must stay in RAM', 'Cloud Run scales to zero between requests, clearing the in-memory routing graph (loaded at startup: ~30–60 sec). Compute Engine keeps the instance alive 24/7, holding OSRM\'s pre-processed road network ready for instant queries.', true);
inputRow(ws3, 'Why OSRM — not Google Maps API', 'Saves ~₱3,100,000 per enrollment', 'Google Maps Distance Matrix API costs $5 per 1,000 pairs. At 1M learners × 10 school detail views = 10M pairs: $50,000 = ₱3,100,000. OSRM is open-source; the only cost is the VM instance.', false);

calcSubHeader(ws3, '  Calculation');
calcColHeaders(ws3);
a3 = false;
calcRow(ws3, 'Instance cost (on-demand)', '$0.0767 / hour', '2,160 hours', '$165.67', '90 days × 24 hrs/day = 2,160 hrs × $0.0767 = $165.67  [Source 7]', a3);
usdSubtotal(ws3, '$165.67');
phpTotalRow(ws3, '$165.67', '₱10,272', '→ Budget tab: "Compute Engine e2-standard-2 (OSRM)" = ₱10,272 / enrollment');
blank(ws3); blank(ws3);

// ── 7. Cloud SQL — 2 vCPU, 7.5 GB RAM ───────────────────────────────────────
sectionTitle(ws3, '  7. Cloud SQL for PostgreSQL — 2 vCPU, 7.5 GB RAM  →  Budget tab: ₱89,280 / year', NC3);
calcSubHeader(ws3, '  Inputs & Assumptions');
inputRow(ws3, 'Instance spec', 'db-custom-2-7680  (2 vCPU, 7.5 GiB RAM)', 'Upgraded from 1 vCPU / 3.75 GiB (staging) to handle concurrent enrollment load. Supports ~200–400 DB writes/second sustained (~20,000–30,000 simultaneous submitting users via PgBouncer).', false);
inputRow(ws3, 'Edition', 'Enterprise — General Purpose', 'Default Cloud SQL edition for new instances. Source: cloud.google.com/sql/pricing [Source 3]', true);
inputRow(ws3, 'vCPU rate (asia-southeast1, on-demand)', '$0.0578 / vCPU / hour', 'Enterprise Edition, General Purpose, Default (on-demand). Source: cloud.google.com/sql/pricing [Source 3]', false);
inputRow(ws3, 'Memory rate (asia-southeast1, on-demand)', '$0.0098 / GiB / hour', 'Enterprise Edition, General Purpose, Default (on-demand). Source: cloud.google.com/sql/pricing [Source 3]', true);
inputRow(ws3, 'SSD storage rate', '$0.222 / GB / month', 'Source: cloud.google.com/sql/pricing [Source 3]', false);
inputRow(ws3, 'Storage provisioned', '50 GB SSD', 'Sufficient for 1M student submissions + school data + audit logs for the pilot year.', true);
inputRow(ws3, 'Runtime', '12 months (always-on)', 'Student submission records must persist year-round — database cannot scale to zero. Contains all enrollment audit logs, ranked choices, and school slot data.', false);

calcSubHeader(ws3, '  Derived Values  (per month, 730 hours)');
inputRow(ws3, 'vCPU cost per month', '$84.39', '2 vCPU × $0.0578/hr × 730 hrs', false);
inputRow(ws3, 'Memory cost per month', '$53.66', '7.5 GiB × $0.0098/GiB-hr × 730 hrs', true);
inputRow(ws3, 'Storage cost per month', '$11.10', '50 GB × $0.222/GB-month', false);
inputRow(ws3, 'Total per month', '$149.15', '$84.39 + $53.66 + $11.10', true);

calcSubHeader(ws3, '  Calculation  [Source 3: cloud.google.com/sql/pricing]');
calcColHeaders(ws3);
a3 = false;
calcRow(ws3, 'vCPU (2 vCPU × $0.0578 × 730 hrs)', '$84.39 / month', '12 months', '$1,012.68', '2 × $0.0578 × 730 = $84.39/month', a3); a3 = !a3;
calcRow(ws3, 'Memory (7.5 GiB × $0.0098 × 730 hrs)', '$53.66 / month', '12 months', '$643.92', '7.5 × $0.0098 × 730 = $53.66/month', a3); a3 = !a3;
calcRow(ws3, 'SSD Storage (50 GB × $0.222)', '$11.10 / month', '12 months', '$133.20', '50 GB × $0.222/GB/month', a3);
usdSubtotal(ws3, '$1,789.80');
phpTotalRow(ws3, '$1,789.80', '₱110,968 → rounded ₱111,000', '→ Budget tab: "Cloud SQL for PostgreSQL (2 vCPU, 7.5 GiB RAM)" = ₱111,000 / year');
blank(ws3); blank(ws3);

// ── 8. Cloud SQL Automated Backups ───────────────────────────────────────────
sectionTitle(ws3, '  8. Cloud SQL Automated Backups (90-day retention)  →  Budget tab: ₱7,440 / year', NC3);
calcSubHeader(ws3, '  Inputs & Assumptions');
inputRow(ws3, 'Backup storage rate', '$0.08 / GB / month', 'Cloud SQL backup storage overage rate. Source: cloud.google.com/sql/pricing [Source 3]', false);
inputRow(ws3, 'Estimated backup storage at steady state', '~110 GB', '90 daily compressed snapshots × ~1 GB each + WAL (write-ahead log) for PITR. Submissions are text-heavy JSON — compress well.', true);
inputRow(ws3, 'Retention policy', '90 days + PITR enabled', 'Required for government enrollment records: covers the full enrollment window plus a post-close period for dispute resolution and audit.', false);

calcSubHeader(ws3, '  Calculation');
calcColHeaders(ws3);
a3 = false;
calcRow(ws3, 'Backup storage (90-day retention)', '$0.08 / GB / month', '~110 GB × 12 months', '~$105.60', '$0.08 × 110 GB = $8.80/month × 12 = $105.60. Rounded to $10/month = $120/year for budget.  [Source 3]', a3);
usdSubtotal(ws3, '~$120.00');
phpTotalRow(ws3, '~$120.00', '₱7,440', '→ Budget tab: "Cloud SQL Automated Backups (90-day retention)" = ₱7,440 / year');
blank(ws3); blank(ws3);

// ── Footer note ───────────────────────────────────────────────────────────────
{
  const r = ws3.addRow(['Items not detailed above (PgBouncer, Cloud Run Jobs, Cloud Storage, BigQuery, Egress/Monitoring) are estimated from official GCP pricing pages at the unit rates shown in the Budget tab Notes column. All rates are subject to change — verify on official Google Cloud pricing pages before DBM submission.']);
  merge(ws3, r.number, 1, r.number, NC3);
  r.height = 32;
  r.getCell(1).font      = { italic: true, size: 9, color: { argb: TEXTGREY } };
  r.getCell(1).alignment = { vertical: 'top', wrapText: true, indent: 1 };
}

// ── Write file ────────────────────────────────────────────────────────────────
wb.xlsx.writeFile(OUT_FILE).then(() => {
  console.log(`✓  Written: ${OUT_FILE}`);
}).catch(err => {
  console.error('Error writing file:', err);
  process.exit(1);
});
