export interface School {
  school_id: string;
  school_name: string;
  latitude: number | null;
  longitude: number | null;

  deped_region: string | null;
  deped_province: string | null;
  deped_municipality: string | null;
  deped_barangay: string | null;

  urban_rural: "U" | "R" | null;
  lgu_income_class: string | null;

  is_esc_participating: boolean;
  school_type: "public" | "private";
  is_huc: boolean | null;

  esc_subsidy_amount: number | null;
  slot_fixed: number | null;
  slot_incentive: number | null;
  slot_additional: number | null;
  slot_total: number | null;
  slot_billed: number | null;
  slot_unutilized: number | null;

  esc_tuition: number | null;
  esc_other_fees: number | null;
  esc_misc_fees: number | null;
  esc_total_fees: number | null;
  esc_net_fees: number | null;
  esc_rating_rank: number | null;
}
