import { useEffect, useMemo, useRef, useState } from "react";
import type { School } from "@/types/school";
import { getFeeRange, getNetFeeRange, getSubsidyRange } from "@/lib/schools";

export type SchoolTypeFilter = "public" | "private";

export interface SchoolFilters {
  search: string;
  schoolTypes: Set<SchoolTypeFilter>;
  escOnly: boolean;
  barangay: string | null;
  feeRange: [number, number];
  subsidyRange: [number, number];
  netFeeRange: [number, number];
}

function inRange(value: number | null, range: [number, number]): boolean {
  if (value === null) return true;
  return value >= range[0] && value <= range[1];
}

export function useSchoolFilters(schools: School[]) {
  const feeBounds = useMemo(() => getFeeRange(schools), [schools]);
  const subsidyBounds = useMemo(() => getSubsidyRange(schools), [schools]);
  const netFeeBounds = useMemo(() => getNetFeeRange(schools), [schools]);

  const [search, setSearch] = useState("");
  const [schoolTypes, setSchoolTypes] = useState<Set<SchoolTypeFilter>>(
    new Set(),
  );
  const [escOnly, setEscOnly] = useState(false);
  const [barangay, setBarangay] = useState<string | null>(null);
  const [feeRange, setFeeRange] = useState<[number, number]>(feeBounds);
  const [subsidyRange, setSubsidyRange] =
    useState<[number, number]>(subsidyBounds);
  const [netFeeRange, setNetFeeRange] =
    useState<[number, number]>(netFeeBounds);

  // `schools` arrives asynchronously (fetched from the live API) and
  // starts empty, so the `useState` initializers above only ever see
  // `[0, 0]` bounds on first mount - `useState`'s initial value isn't
  // re-evaluated on later renders. Sync the range state once, the
  // first time real bounds appear, without touching it again (so a
  // user's own slider adjustment afterward is never overwritten).
  const boundsAppliedRef = useRef(false);
  useEffect(() => {
    if (boundsAppliedRef.current || schools.length === 0) return;
    boundsAppliedRef.current = true;
    setFeeRange(feeBounds);
    setSubsidyRange(subsidyBounds);
    setNetFeeRange(netFeeBounds);
  }, [schools, feeBounds, subsidyBounds, netFeeBounds]);

  const filteredSchools = useMemo(() => {
    const query = search.trim().toLowerCase();
    return schools.filter((school) => {
      if (query && !school.school_name.toLowerCase().includes(query)) {
        return false;
      }
      if (schoolTypes.size > 0 && !schoolTypes.has(school.school_type)) {
        return false;
      }
      if (escOnly && !school.is_esc_participating) {
        return false;
      }
      if (barangay && school.barangay !== barangay) {
        return false;
      }
      if (!inRange(school.esc_total_fees, feeRange)) return false;
      if (!inRange(school.esc_subsidy_amount, subsidyRange)) return false;
      if (!inRange(school.esc_net_fees, netFeeRange)) return false;
      return true;
    });
  }, [
    schools,
    search,
    schoolTypes,
    escOnly,
    barangay,
    feeRange,
    subsidyRange,
    netFeeRange,
  ]);

  const toggleSchoolType = (type: SchoolTypeFilter) => {
    setSchoolTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const resetFilters = () => {
    setSearch("");
    setSchoolTypes(new Set());
    setEscOnly(false);
    setBarangay(null);
    setFeeRange(feeBounds);
    setSubsidyRange(subsidyBounds);
    setNetFeeRange(netFeeBounds);
  };

  return {
    filters: {
      search,
      schoolTypes,
      escOnly,
      barangay,
      feeRange,
      subsidyRange,
      netFeeRange,
    },
    feeBounds,
    subsidyBounds,
    netFeeBounds,
    filteredSchools,
    setSearch,
    toggleSchoolType,
    setEscOnly,
    setBarangay,
    setFeeRange,
    setSubsidyRange,
    setNetFeeRange,
    resetFilters,
  };
}
