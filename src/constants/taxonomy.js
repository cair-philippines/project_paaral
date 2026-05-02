export const SCHOOL_TYPES = {
  public_es:       "Public Elementary",
  private_es:      "Private Elementary",
  public_jhs:      "Public JHS",
  private_jhs:     "Private JHS (No ESC)",
  private_jhs_esc: "Private JHS (With ESC)",
}

export const REGIONS = {
  ncr: "National Capital Region (NCR)",
  iva: "Region IV-A (CALABARZON)",
}

// city_type is the ESC subsidy tier key — set on every school record.
// ncr   → all NCR schools (₱13,000)
// huc   → Lucena City only, sole IVA HUC per 2025 PSA list (₱11,000)
// other → all remaining Region IV-A schools (₱9,000)
export const CITY_TYPES = {
  ncr:   { label: "NCR",                   subsidy: 13000 },
  huc:   { label: "Highly Urbanized City", subsidy: 11000 },
  other: { label: "Other",                 subsidy: 9000  },
}

export const getSubsidy = (city_type) => CITY_TYPES[city_type]?.subsidy ?? 0
