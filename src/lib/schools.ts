import type { School } from "@/types/school";
import qcSchools from "@/lib/data/qc-schools.json";
import { apiGet } from "@/lib/api";

/** Build-time-only snapshot, used solely so the static export knows
 * which `/schools/[school_id]` pages to pre-render (`generateStaticParams`
 * can't call a live API at build time without making the build itself
 * depend on the backend being reachable). The live browse/search
 * experience uses `fetchSchools()` instead — see `docs/post-pilot-scaling.md`
 * for the tradeoff this snapshot approach accepts. */
export function getQcSchools(): School[] {
  return qcSchools as School[];
}

/** Wire shape of `GET /api/v1/schools` — camelCase, matching the
 * backend's `CamelModel` convention (same as the auth endpoint's
 * `LoginLookupResult`). Kept separate from `School`, which stays
 * snake_case to match the many already-existing components built
 * against the original BigQuery-column-named static JSON - translating
 * at this one boundary is far less invasive than renaming every field
 * access across the app. */
interface ApiSchool {
  schoolId: string;
  schoolName: string;
  latitude: number | null;
  longitude: number | null;
  region: string | null;
  province: string | null;
  municipality: string | null;
  barangay: string | null;
  urbanRural: "U" | "R" | null;
  lguIncomeClass: string | null;
  isEscParticipating: boolean;
  schoolType: "public" | "private";
  isHuc: boolean | null;
  escSubsidyAmount: number | null;
  slotTotal: number | null;
  slotUnutilized: number | null;
  escTuition: number | null;
  escOtherFees: number | null;
  escMiscFees: number | null;
  escTotalFees: number | null;
  escNetFees: number | null;
  escRatingRank: number | null;
}

function mapApiSchool(s: ApiSchool): School {
  return {
    school_id: s.schoolId,
    school_name: s.schoolName,
    latitude: s.latitude,
    longitude: s.longitude,
    region: s.region,
    province: s.province,
    municipality: s.municipality,
    barangay: s.barangay,
    urban_rural: s.urbanRural,
    lgu_income_class: s.lguIncomeClass,
    is_esc_participating: s.isEscParticipating,
    school_type: s.schoolType,
    is_huc: s.isHuc,
    esc_subsidy_amount: s.escSubsidyAmount,
    slot_total: s.slotTotal,
    slot_unutilized: s.slotUnutilized,
    esc_tuition: s.escTuition,
    esc_other_fees: s.escOtherFees,
    esc_misc_fees: s.escMiscFees,
    esc_total_fees: s.escTotalFees,
    esc_net_fees: s.escNetFees,
    esc_rating_rank: s.escRatingRank,
  };
}

/** Live school list from `paaral-student-api` (Chunk 16, step 7) —
 * the runtime data source for `/browse`, replacing `getQcSchools()`.
 * Called with no filters; `useSchoolFilters` still does the actual
 * filtering client-side (Option A, see `docs/post-pilot-scaling.md`). */
export async function fetchSchools(): Promise<School[]> {
  const schools = await apiGet<ApiSchool[]>("/api/v1/schools");
  return schools.map(mapApiSchool);
}

export function getSchoolById(id: string): School | undefined {
  return getQcSchools().find((school) => school.school_id === id);
}

export function getBarangayOptions(schools: School[]): string[] {
  const barangays = new Set<string>();
  for (const school of schools) {
    if (school.barangay) barangays.add(school.barangay);
  }
  return Array.from(barangays).sort();
}

function getFieldRange(
  schools: School[],
  field: "esc_total_fees" | "esc_net_fees" | "esc_subsidy_amount"
): [number, number] {
  const values = schools
    .map((school) => school[field])
    .filter((value): value is number => value !== null);
  if (values.length === 0) return [0, 0];
  return [Math.min(...values), Math.max(...values)];
}

export function getFeeRange(schools: School[]): [number, number] {
  return getFieldRange(schools, "esc_total_fees");
}

export function getNetFeeRange(schools: School[]): [number, number] {
  return getFieldRange(schools, "esc_net_fees");
}

export function getSubsidyRange(schools: School[]): [number, number] {
  return getFieldRange(schools, "esc_subsidy_amount");
}

export function pesos(value: number | null): string {
  if (value === null) return "Not available";
  if (value === 0) return "Free";
  return new Intl.NumberFormat("en-PH", {
    style: "currency",
    currency: "PHP",
    maximumFractionDigits: 0,
  }).format(value);
}

export function netFeesLabel(value: number | null): string {
  if (value === null) return "Not available";
  if (value <= 0) return "Fully covered by subsidy";
  return pesos(value);
}

export function titleCase(value: string | null): string | null {
  if (value === null) return null;
  return value
    .toLowerCase()
    .split(" ")
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : word))
    .join(" ");
}

export interface TypeBadge {
  label: string;
  className: string;
}

export function getTypeBadge(school: School): TypeBadge {
  if (school.school_type === "public") {
    return { label: "Public", className: "bg-primary/10 text-primary" };
  }
  if (school.is_esc_participating) {
    return {
      label: "Private, ESC-participating",
      className: "bg-[#16a34a]/10 text-[#16a34a]",
    };
  }
  return {
    label: "Private, no ESC",
    className: "bg-[#f59e0b]/10 text-[#f59e0b]",
  };
}

export interface SlotAvailability {
  total: number | null;
  available: number | null;
  tone: "green" | "amber" | "red" | "unknown";
}

export function getSlotAvailability(school: School): SlotAvailability {
  const { slot_total, slot_unutilized } = school;
  if (slot_total === null || slot_unutilized === null) {
    return { total: slot_total, available: null, tone: "unknown" };
  }

  const available = Math.max(slot_unutilized, 0);
  const ratio = slot_total > 0 ? available / slot_total : 0;

  let tone: SlotAvailability["tone"] = "green";
  if (slot_unutilized <= 0) tone = "red";
  else if (ratio <= 0.2) tone = "amber";

  return { total: slot_total, available, tone };
}
