import type { School } from "@/types/school";
import qcSchools from "@/lib/data/qc-schools.json";

export function getQcSchools(): School[] {
  return qcSchools as School[];
}

export function getSchoolById(id: string): School | undefined {
  return getQcSchools().find((school) => school.school_id === id);
}

export function getBarangayOptions(schools: School[]): string[] {
  const barangays = new Set<string>();
  for (const school of schools) {
    if (school.deped_barangay) barangays.add(school.deped_barangay);
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
