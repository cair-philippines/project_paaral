/**
 * PLACEHOLDER — not a real DepEd-published date. Set by Paula 2026-08-20
 * as a stand-in deadline for the landing page hero until a real ESC
 * application-period date is confirmed. Update when that's available.
 */
export const ESC_APPLICATION_DEADLINE = new Date("2027-05-30");

export const ESC_APPLICATION_DEADLINE_LABEL = "May 30, 2027";

export const APPLICATION_STORAGE_KEY = "paaral_v3_account";

/** Demo DepEd ICTS SSO stand-ins — see CLAUDE.md "Demo Credentials". Three
 * LRNs, three distinct login behaviors: fresh, with-a-pending-draft, and
 * invalid (anything else typed into the login modal falls through to the
 * generic "not found" error — it doesn't need its own constant). */
export const TEST_LRN = "100000000001";
export const TEST_EMAIL = `${TEST_LRN}@deped.gov.ph`;

export const TEST_LRN_WITH_DRAFT = "100000000002";
export const TEST_EMAIL_WITH_DRAFT = `${TEST_LRN_WITH_DRAFT}@deped.gov.ph`;

export const LEARNER_RECORD = {
  firstName: "Juan",
  mi: "M",
  lastName: "dela Cruz",
  school: "Bagumbayan Elementary School",
  grade: "Grade 6",
  municipality: "Taguig City, Metro Manila",
  division: "Division of Taguig-Pateros",
};

/**
 * Pre-loaded draft wishlist for LRN 100000000002, re-picked against the real
 * Quezon City dataset (the original mockup's 3 draft schools — Bagumbayan
 * NHS, St. Mary's Academy of Taguig, Pasig Grace Christian School — don't
 * exist in this pilot's QC-only data). Kept the same 1-public/2-private
 * shape: a public NHS plus two ESC-participating private schools with real
 * fee and slot data so the application panel has something meaningful to
 * show.
 */
export const DRAFT_WISHLIST_SCHOOL_IDS = ["305330", "406444", "406467"];
