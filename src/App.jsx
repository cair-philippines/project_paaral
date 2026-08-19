import React, { useMemo, useState, useEffect, useRef, useCallback } from "react";
import Map, { Source, Layer, Marker, NavigationControl } from 'react-map-gl/mapbox';
import 'mapbox-gl/dist/mapbox-gl.css';
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Clock3,
  Info,
  Layers,
  MapPin,
  Minus,
  Plus,
  Search,
  SlidersHorizontal,
  Sparkles,
  Star,
  X,
  Heart,
  ShieldCheck,
  SendHorizontal,
  Navigation,
  GripVertical,
  LogOut,
  Award,
  FileCheck,
  AlertCircle,
  RefreshCw,
  User,
  Menu
} from "lucide-react";

// --- SYNTHETIC DATA (Expanded to 50 Schools) ---
const schools = [
  { id: "SCH001", name: "St. Mary's Academy of Taguig", type: "private_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Taguig City", barangay: "Bagumbayan", postal_code: "1630", lat: 14.5176, lng: 121.0509, tuition: 45000, esc_subsidy: 13000, net_cost: 32000, slots_total: 40, slots_available: 12, distance_km: 3.2, commute_minutes: 15, esc_rating: 4, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH002", name: "Bagumbayan National High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Taguig City", barangay: "Bagumbayan", postal_code: "1630", lat: 14.5211, lng: 121.0576, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 360, slots_available: 22, distance_km: 1.4, commute_minutes: 6, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH003", name: "Senator Renato Cayetano Memorial Science and Technology High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Taguig City", barangay: "Ususan", postal_code: "1639", lat: 14.5382, lng: 121.0675, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 280, slots_available: 9, distance_km: 4.8, commute_minutes: 22, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH004", name: "Pateros Catholic School", type: "private_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Pateros", barangay: "San Roque", postal_code: "1620", lat: 14.5455, lng: 121.0699, tuition: 52000, esc_subsidy: 13000, net_cost: 39000, slots_total: 45, slots_available: 18, distance_km: 5.7, commute_minutes: 28, esc_rating: 5, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH005", name: "Fort Bonifacio Christian Academy", type: "private_esc", sector: "non_sectarian", region: "NCR", province: "Metro Manila", municipality: "Makati City", barangay: "Cembo", postal_code: "1214", lat: 14.5538, lng: 121.0436, tuition: 47000, esc_subsidy: 13000, net_cost: 34000, slots_total: 42, slots_available: 17, distance_km: 7.1, commute_minutes: 31, esc_rating: 4, religious_affiliation: "Non-Sectarian", admission_category: "ESC Partner" },
  { id: "SCH006", name: "Pasig Grace Christian School", type: "private_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Pasig City", barangay: "Maybunga", postal_code: "1607", lat: 14.5752, lng: 121.0837, tuition: 42000, esc_subsidy: 13000, net_cost: 29000, slots_total: 50, slots_available: 15, distance_km: 10.8, commute_minutes: 42, esc_rating: 4, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH007", name: "St. Paul College Pasig", type: "private_no_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Pasig City", barangay: "Ugong", postal_code: "1604", lat: 14.5845, lng: 121.0797, tuition: 118000, esc_subsidy: 0, net_cost: 118000, slots_total: 60, slots_available: 21, distance_km: 11.6, commute_minutes: 45, esc_rating: 0, religious_affiliation: "Sectarian", admission_category: "Highly Selective" },
  { id: "SCH008", name: "Imus National High School", type: "public", sector: null, region: "Region IV-A", province: "Cavite", municipality: "Imus City", barangay: "Poblacion", postal_code: "4103", lat: 14.4297, lng: 120.9367, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 420, slots_available: 44, distance_km: 21.4, commute_minutes: 58, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH009", name: "St. Edward Integrated School", type: "private_esc", sector: "sectarian", region: "Region IV-A", province: "Cavite", municipality: "Imus City", barangay: "Buhay na Tubig", postal_code: "4103", lat: 14.4144, lng: 120.9577, tuition: 62000, esc_subsidy: 11000, net_cost: 51000, slots_total: 55, slots_available: 27, distance_km: 24.6, commute_minutes: 62, esc_rating: 4, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH010", name: "Dasmarinas Integrated High School", type: "public", sector: null, region: "Region IV-A", province: "Cavite", municipality: "Dasmarinas City", barangay: "Zone IV", postal_code: "4114", lat: 14.3294, lng: 120.9366, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 500, slots_available: 63, distance_km: 34.9, commute_minutes: 78, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH011", name: "Cavite Christian School", type: "private_esc", sector: "sectarian", region: "Region IV-A", province: "Cavite", municipality: "Bacoor City", barangay: "Molino III", postal_code: "4102", lat: 14.4117, lng: 120.9742, tuition: 38000, esc_subsidy: 11000, net_cost: 27000, slots_total: 48, slots_available: 8, distance_km: 22.2, commute_minutes: 54, esc_rating: 3, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH012", name: "Southville International School Cavite", type: "private_no_esc", sector: "non_sectarian", region: "Region IV-A", province: "Cavite", municipality: "Bacoor City", barangay: "Habitat", postal_code: "4102", lat: 14.4338, lng: 120.9643, tuition: 142000, esc_subsidy: 0, net_cost: 142000, slots_total: 35, slots_available: 19, distance_km: 20.5, commute_minutes: 49, esc_rating: 0, religious_affiliation: "Non-Sectarian", admission_category: "Highly Selective" },
  { id: "SCH013", name: "San Pedro Relocation Center National High School", type: "public", sector: null, region: "Region IV-A", province: "Laguna", municipality: "San Pedro City", barangay: "Landayan", postal_code: "4023", lat: 14.3588, lng: 121.0536, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 390, slots_available: 17, distance_km: 25.7, commute_minutes: 64, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH014", name: "Binan City Science and Technology High School", type: "public", sector: null, region: "Region IV-A", province: "Laguna", municipality: "Binan City", barangay: "San Antonio", postal_code: "4024", lat: 14.3371, lng: 121.0804, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 260, slots_available: 12, distance_km: 30.6, commute_minutes: 72, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH015", name: "Colegio de San Juan de Letran Calamba", type: "private_esc", sector: "sectarian", region: "Region IV-A", province: "Laguna", municipality: "Calamba City", barangay: "Bucal", postal_code: "4027", lat: 14.1981, lng: 121.1653, tuition: 59000, esc_subsidy: 11000, net_cost: 48000, slots_total: 70, slots_available: 31, distance_km: 53.8, commute_minutes: 96, esc_rating: 5, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH016", name: "Laguna BelAir School", type: "private_no_esc", sector: "non_sectarian", region: "Region IV-A", province: "Laguna", municipality: "Santa Rosa City", barangay: "Don Jose", postal_code: "4026", lat: 14.2825, lng: 121.0894, tuition: 98000, esc_subsidy: 0, net_cost: 98000, slots_total: 45, slots_available: 23, distance_km: 39.8, commute_minutes: 83, esc_rating: 0, religious_affiliation: "Non-Sectarian", admission_category: "Highly Selective" },
  { id: "SCH017", name: "Malolos Integrated School", type: "public", sector: null, region: "Region III", province: "Bulacan", municipality: "Malolos City", barangay: "Guinhawa", postal_code: "3000", lat: 14.8527, lng: 120.816, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 430, slots_available: 51, distance_km: 48.2, commute_minutes: 92, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH018", name: "Meycauayan National High School", type: "public", sector: null, region: "Region III", province: "Bulacan", municipality: "Meycauayan City", barangay: "Calvario", postal_code: "3020", lat: 14.7368, lng: 120.9608, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 470, slots_available: 28, distance_km: 32.7, commute_minutes: 75, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH019", name: "St. Anne's Catholic School of Bulacan", type: "private_esc", sector: "sectarian", region: "Region III", province: "Bulacan", municipality: "Malolos City", barangay: "Catmon", postal_code: "3000", lat: 14.8471, lng: 120.8111, tuition: 41000, esc_subsidy: 9000, net_cost: 32000, slots_total: 52, slots_available: 16, distance_km: 49.1, commute_minutes: 95, esc_rating: 4, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH020", name: "Bulacan Ecumenical School", type: "private_esc", sector: "non_sectarian", region: "Region III", province: "Bulacan", municipality: "Marilao", barangay: "Loma de Gato", postal_code: "3019", lat: 14.7571, lng: 120.9488, tuition: 36000, esc_subsidy: 9000, net_cost: 27000, slots_total: 40, slots_available: 4, distance_km: 35.4, commute_minutes: 81, esc_rating: 3, religious_affiliation: "Non-Sectarian", admission_category: "ESC Partner" },
  { id: "SCH021", name: "Our Lady of Guadalupe School San Jose del Monte", type: "private_no_esc", sector: "sectarian", region: "Region III", province: "Bulacan", municipality: "San Jose del Monte City", barangay: "Tungkong Mangga", postal_code: "3023", lat: 14.8167, lng: 121.0754, tuition: 74000, esc_subsidy: 0, net_cost: 74000, slots_total: 44, slots_available: 11, distance_km: 43.9, commute_minutes: 98, esc_rating: 0, religious_affiliation: "Sectarian", admission_category: "Highly Selective" },
  { id: "SCH022", name: "Angeles City National Trade School", type: "public", sector: null, region: "Region III", province: "Pampanga", municipality: "Angeles City", barangay: "Pulungbulu", postal_code: "2009", lat: 15.1456, lng: 120.5881, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 520, slots_available: 76, distance_km: 91.5, commute_minutes: 132, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH023", name: "San Fernando High School", type: "public", sector: null, region: "Region III", province: "Pampanga", municipality: "City of San Fernando", barangay: "Dolores", postal_code: "2000", lat: 15.0287, lng: 120.6893, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 490, slots_available: 33, distance_km: 78.8, commute_minutes: 118, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH024", name: "Holy Angel Academy", type: "private_esc", sector: "sectarian", region: "Region III", province: "Pampanga", municipality: "Angeles City", barangay: "Sto. Rosario", postal_code: "2009", lat: 15.1344, lng: 120.5906, tuition: 48000, esc_subsidy: 9000, net_cost: 39000, slots_total: 65, slots_available: 26, distance_km: 90.7, commute_minutes: 130, esc_rating: 5, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH025", name: "Pampanga Central Institute", type: "private_esc", sector: "non_sectarian", region: "Region III", province: "Pampanga", municipality: "Mexico", barangay: "San Antonio", postal_code: "2021", lat: 15.0701, lng: 120.7219, tuition: 35000, esc_subsidy: 9000, net_cost: 26000, slots_total: 48, slots_available: 14, distance_km: 82.2, commute_minutes: 124, esc_rating: 4, religious_affiliation: "Non-Sectarian", admission_category: "ESC Partner" },
  { id: "SCH026", name: "Clarkfield Learning Center", type: "private_no_esc", sector: "non_sectarian", region: "Region III", province: "Pampanga", municipality: "Mabalacat City", barangay: "Dau", postal_code: "2010", lat: 15.1842, lng: 120.5939, tuition: 128000, esc_subsidy: 0, net_cost: 128000, slots_total: 38, slots_available: 20, distance_km: 95.8, commute_minutes: 140, esc_rating: 0, religious_affiliation: "Non-Sectarian", admission_category: "Highly Selective" },
  { id: "SCH027", name: "Rizal High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Pasig City", barangay: "Caniogan", postal_code: "1606", lat: 14.5670, lng: 121.0773, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 1200, slots_available: 45, distance_km: 12.1, commute_minutes: 48, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH028", name: "Makati Science High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Makati City", barangay: "Cembo", postal_code: "1214", lat: 14.5582, lng: 121.0543, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 300, slots_available: 5, distance_km: 6.5, commute_minutes: 25, esc_rating: 0, religious_affiliation: "Public", admission_category: "Highly Selective" },
  { id: "SCH029", name: "Pitogo High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Makati City", barangay: "Pitogo", postal_code: "1213", lat: 14.5555, lng: 121.0478, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 400, slots_available: 50, distance_km: 6.8, commute_minutes: 28, esc_rating: 0, religious_affiliation: "Public", admission_category: "Open Admission" },
  { id: "SCH030", name: "Ateneo de Manila Junior High School", type: "private_no_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Quezon City", barangay: "Loyola Heights", postal_code: "1108", lat: 14.6394, lng: 121.0781, tuition: 180000, esc_subsidy: 0, net_cost: 180000, slots_total: 400, slots_available: 15, distance_km: 14.5, commute_minutes: 55, esc_rating: 0, religious_affiliation: "Sectarian", admission_category: "Highly Selective" },
  { id: "SCH031", name: "Miriam College Middle School", type: "private_no_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Quezon City", barangay: "UP Campus", postal_code: "1101", lat: 14.6465, lng: 121.0754, tuition: 165000, esc_subsidy: 0, net_cost: 165000, slots_total: 250, slots_available: 20, distance_km: 15.2, commute_minutes: 58, esc_rating: 0, religious_affiliation: "Sectarian", admission_category: "Highly Selective" },
  { id: "SCH032", name: "Quezon City Science High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Quezon City", barangay: "Bago Bantay", postal_code: "1105", lat: 14.6542, lng: 121.0298, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 350, slots_available: 0, distance_km: 17.1, commute_minutes: 65, esc_rating: 0, religious_affiliation: "Public", admission_category: "Highly Selective" },
  { id: "SCH033", name: "San Francisco High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Quezon City", barangay: "Bago Bantay", postal_code: "1105", lat: 14.6552, lng: 121.0312, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 800, slots_available: 120, distance_km: 17.3, commute_minutes: 66, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH034", name: "St. Bridget School", type: "private_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Quezon City", barangay: "Loyola Heights", postal_code: "1108", lat: 14.6321, lng: 121.0715, tuition: 85000, esc_subsidy: 13000, net_cost: 72000, slots_total: 120, slots_available: 35, distance_km: 14.0, commute_minutes: 52, esc_rating: 4, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH035", name: "Manila Science High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Manila", barangay: "Ermita", postal_code: "1000", lat: 14.5866, lng: 120.9856, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 250, slots_available: 10, distance_km: 10.5, commute_minutes: 40, esc_rating: 0, religious_affiliation: "Public", admission_category: "Highly Selective" },
  { id: "SCH036", name: "Araullo High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Manila", barangay: "Ermita", postal_code: "1000", lat: 14.5821, lng: 120.9835, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 900, slots_available: 85, distance_km: 10.2, commute_minutes: 38, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH037", name: "St. Scholastica's College Manila", type: "private_no_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Manila", barangay: "Malate", postal_code: "1004", lat: 14.5636, lng: 120.9947, tuition: 135000, esc_subsidy: 0, net_cost: 135000, slots_total: 200, slots_available: 40, distance_km: 8.5, commute_minutes: 35, esc_rating: 0, religious_affiliation: "Sectarian", admission_category: "Highly Selective" },
  { id: "SCH038", name: "De La Salle Santiago Zobel School", type: "private_no_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Muntinlupa", barangay: "Ayala Alabang", postal_code: "1780", lat: 14.4239, lng: 121.0189, tuition: 195000, esc_subsidy: 0, net_cost: 195000, slots_total: 300, slots_available: 25, distance_km: 18.5, commute_minutes: 60, esc_rating: 0, religious_affiliation: "Sectarian", admission_category: "Highly Selective" },
  { id: "SCH039", name: "Muntinlupa National High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Muntinlupa", barangay: "Poblacion", postal_code: "1776", lat: 14.3812, lng: 121.0335, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 1500, slots_available: 210, distance_km: 21.0, commute_minutes: 65, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH040", name: "San Beda College Alabang", type: "private_no_esc", sector: "sectarian", region: "NCR", province: "Metro Manila", municipality: "Muntinlupa", barangay: "Alabang", postal_code: "1780", lat: 14.4285, lng: 121.0267, tuition: 155000, esc_subsidy: 0, net_cost: 155000, slots_total: 350, slots_available: 45, distance_km: 17.5, commute_minutes: 55, esc_rating: 0, religious_affiliation: "Sectarian", admission_category: "Highly Selective" },
  { id: "SCH041", name: "Muntinlupa Science High School", type: "public", sector: null, region: "NCR", province: "Metro Manila", municipality: "Muntinlupa", barangay: "Tunasan", postal_code: "1773", lat: 14.3756, lng: 121.0421, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 200, slots_available: 0, distance_km: 22.5, commute_minutes: 70, esc_rating: 0, religious_affiliation: "Public", admission_category: "Highly Selective" },
  { id: "SCH042", name: "Carmona National High School", type: "public", sector: null, region: "Region IV-A", province: "Cavite", municipality: "Carmona", barangay: "Maduya", postal_code: "4116", lat: 14.3167, lng: 121.0567, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 600, slots_available: 80, distance_km: 27.5, commute_minutes: 75, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH043", name: "Biñan National High School", type: "public", sector: null, region: "Region IV-A", province: "Laguna", municipality: "Binan City", barangay: "Santo Tomas", postal_code: "4024", lat: 14.3312, lng: 121.0845, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 1100, slots_available: 150, distance_km: 29.8, commute_minutes: 80, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH044", name: "La Consolacion College Biñan", type: "private_esc", sector: "sectarian", region: "Region IV-A", province: "Laguna", municipality: "Binan City", barangay: "Santo Tomas", postal_code: "4024", lat: 14.3298, lng: 121.0855, tuition: 55000, esc_subsidy: 11000, net_cost: 44000, slots_total: 180, slots_available: 40, distance_km: 30.0, commute_minutes: 82, esc_rating: 4, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH045", name: "Santa Rosa Science and Technology High School", type: "public", sector: null, region: "Region IV-A", province: "Laguna", municipality: "Santa Rosa City", barangay: "Market Area", postal_code: "4026", lat: 14.3111, lng: 121.1111, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 250, slots_available: 12, distance_km: 34.5, commute_minutes: 90, esc_rating: 0, religious_affiliation: "Public", admission_category: "Highly Selective" },
  { id: "SCH046", name: "Bulacan State University Laboratory High School", type: "public", sector: null, region: "Region III", province: "Bulacan", municipality: "Malolos City", barangay: "Guinhawa", postal_code: "3000", lat: 14.8580, lng: 120.8145, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 300, slots_available: 5, distance_km: 49.0, commute_minutes: 95, esc_rating: 0, religious_affiliation: "Public", admission_category: "Highly Selective" },
  { id: "SCH047", name: "Marcelo H. Del Pilar National High School", type: "public", sector: null, region: "Region III", province: "Bulacan", municipality: "Malolos City", barangay: "Santa Isabel", postal_code: "3000", lat: 14.8465, lng: 120.8123, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 1800, slots_available: 300, distance_km: 48.5, commute_minutes: 93, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH048", name: "Bocaue National High School", type: "public", sector: null, region: "Region III", province: "Bulacan", municipality: "Bocaue", barangay: "Igulot", postal_code: "3018", lat: 14.7955, lng: 120.9321, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 1200, slots_available: 150, distance_km: 41.2, commute_minutes: 85, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" },
  { id: "SCH049", name: "St. Paul College of Bocaue", type: "private_esc", sector: "sectarian", region: "Region III", province: "Bulacan", municipality: "Bocaue", barangay: "Wakas", postal_code: "3018", lat: 14.7988, lng: 120.9255, tuition: 45000, esc_subsidy: 9000, net_cost: 36000, slots_total: 150, slots_available: 35, distance_km: 42.0, commute_minutes: 87, esc_rating: 3, religious_affiliation: "Sectarian", admission_category: "ESC Partner" },
  { id: "SCH050", name: "Pampanga High School", type: "public", sector: null, region: "Region III", province: "Pampanga", municipality: "City of San Fernando", barangay: "Lourdes", postal_code: "2000", lat: 15.0321, lng: 120.6821, tuition: 0, esc_subsidy: 0, net_cost: 0, slots_total: 2000, slots_available: 450, distance_km: 79.5, commute_minutes: 120, esc_rating: 0, religious_affiliation: "Public", admission_category: "Priority Decongestion" }
];

const typeMeta = {
  public: { label: "Public", badge: "bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white", dot: "#60a5fa" },
  private_esc: { label: "Private with ESC", badge: "bg-[#16a34a] text-white", dot: "#4ade80" },
  private_no_esc: { label: "Private no ESC", badge: "bg-[#f59e0b] text-white", dot: "#fbbf24" },
};

const regionOptions = ["NCR", "Region III", "Region IV-A"];
const provinceOptions = ["Metro Manila", "Cavite", "Laguna", "Bulacan", "Pampanga"];
const municipalityOptions = [
  "Taguig City", "Pateros", "Makati City", "Pasig City", "Imus City", "Bacoor City",
  "Dasmarinas City", "San Pedro City", "Binan City", "Santa Rosa City", "Calamba City",
  "Meycauayan City", "Malolos City", "Marilao", "San Jose del Monte City", "Angeles City",
  "City of San Fernando", "Mexico", "Mabalacat City", "Quezon City", "Manila", "Muntinlupa",
  "Carmona", "Bocaue"
];
const barangayOptions = [
  "Bagumbayan", "Ususan", "San Roque", "Cembo", "Maybunga", "Ugong", "Poblacion",
  "Buhay na Tubig", "Zone IV", "Molino III", "Habitat", "Landayan", "San Antonio",
  "Bucal", "Don Jose", "Guinhawa", "Calvario", "Catmon", "Loma de Gato", "Tungkong Mangga",
  "Pulungbulu", "Dolores", "Sto. Rosario", "Dau", "Caniogan", "Pitogo", "Loyola Heights",
  "UP Campus", "Bago Bantay", "Ermita", "Malate", "Ayala Alabang", "Tunasan", "Maduya",
  "Santo Tomas", "Market Area", "Santa Isabel", "Igulot", "Wakas", "Lourdes"
];

const pesos = (value) =>
  value === 0 ? "Free" : new Intl.NumberFormat("en-PH", { style: "currency", currency: "PHP", maximumFractionDigits: 0 }).format(value);

const pct = (available, total) => Math.round((available / total) * 100);

const schoolTypeMatches = (school, selectedTypes) => {
  if (!selectedTypes.length) return true;
  return selectedTypes.includes(school.type);
};

const commuteBucketMatches = (school, buckets) => {
  if (!buckets.length) return true;
  return buckets.some((bucket) => {
    if (bucket === "under5") return school.commute_minutes < 5;
    if (bucket === "15to30") return school.commute_minutes >= 15 && school.commute_minutes <= 30;
    if (bucket === "over30") return school.commute_minutes > 30;
    return true;
  });
};

const slotTone = (school) => {
  const ratio = pct(school.slots_available, school.slots_total);
  if (school.slots_available === 0 || ratio <= 5) return "bg-[#dc2626]";
  if (ratio <= 20) return "bg-[#f59e0b]";
  return "bg-[#16a34a]";
};

const accordionDefaults = { location: true, distance: true, commute: true, tuition: true, subsidy: true, type: true };

// --- MAP GEOMETRY UTILS (Carto Light Styling) ---
const MAP_BOUNDS = { minLng: 120.52, maxLng: 121.25, minLat: 14.15, maxLat: 15.22 };
const MAP_WIDTH = 750;
const MAP_HEIGHT = 1000;

const WATER_COLOR = "#d1dce5"; 
const LAND_COLOR = "#f0f3f4";  
const HIGHWAY_COLOR = "#ffffff";
const HIGHWAY_BORDER = "#e2e8f0";

const project = (lat, lng) => {
  return {
    x: ((lng - MAP_BOUNDS.minLng) / (MAP_BOUNDS.maxLng - MAP_BOUNDS.minLng)) * MAP_WIDTH,
    y: MAP_HEIGHT - ((lat - MAP_BOUNDS.minLat) / (MAP_BOUNDS.maxLat - MAP_BOUNDS.minLat)) * MAP_HEIGHT,
  };
};

const projectPoint = (school) => project(school.lat, school.lng);

function MapPolygon({ points, fill, stroke, strokeWidth = "2", opacity = "1" }) {
  const projectedPoints = points.map(([lat, lng]) => `${project(lat, lng).x},${project(lat, lng).y}`).join(" ");
  return <polygon points={projectedPoints} fill={fill} stroke={stroke} strokeWidth={strokeWidth} opacity={opacity} />;
}

function MapPath({ points, stroke, strokeWidth, fill = "none", dasharray = "none", opacity = "1" }) {
  const d = points.map(([lat, lng], i) => `${i === 0 ? "M" : "L"} ${project(lat, lng).x} ${project(lat, lng).y}`).join(" ");
  return <path d={d} fill={fill} stroke={stroke} strokeWidth={strokeWidth} strokeDasharray={dasharray} strokeLinecap="round" strokeLinejoin="round" opacity={opacity} />;
}

function MapLabel({ lat, lng, text, fontSize = "13", fontWeight = "bold", opacity = "0.6" }) {
  const coords = project(lat, lng);
  return <text x={coords.x} y={coords.y} fill="#64748b" fontSize={fontSize} fontWeight={fontWeight} fontFamily="sans-serif" opacity={opacity} textAnchor="middle" letterSpacing="0.05em">{text}</text>;
}

const MANILA_BAY_POINTS = [[14.2, 120.52], [14.25, 120.72], [14.3, 120.8], [14.38, 120.86], [14.45, 120.92], [14.48, 120.9], [14.45, 120.95], [14.48, 120.98], [14.54, 120.99], [14.6, 120.96], [14.68, 120.94], [14.76, 120.88], [14.8, 120.75], [14.86, 120.52]];
const LAGUNA_LAKE_POINTS = [[14.53, 121.1], [14.45, 121.05], [14.35, 121.06], [14.28, 121.12], [14.2, 121.18], [14.2, 121.25], [14.5, 121.25], [14.5, 121.18]];
const TAAL_LAKE_POINTS = [[14.08, 120.94], [14.05, 121.02], [14.02, 121.1], [13.94, 121.08], [13.92, 120.99], [13.98, 120.92]];

const EDSA_POINTS = [[14.54, 120.99], [14.54, 121.02], [14.56, 121.04], [14.58, 121.06], [14.62, 121.05], [14.65, 121.03], [14.66, 120.99]];
const SLEX_POINTS = [[14.54, 121.02], [14.48, 121.04], [14.42, 121.04], [14.32, 121.08], [14.25, 121.1], [14.2, 121.13]];
const NLEX_POINTS = [[14.66, 120.99], [14.7, 120.98], [14.75, 120.95], [14.82, 120.88], [14.95, 120.78], [15.05, 120.7], [15.18, 120.59]];

// --- UI COMPONENTS ---
function FilterSection({ title, id, open, onToggle, children }) {
  return (
    <div className="border-b border-[#e2e4e9] py-4 last:border-b-0">
      <button type="button" onClick={() => onToggle(id)} className="flex w-full items-center justify-between text-left focus:outline-none">
        <span className="text-sm font-semibold text-[#1a1d23]">{title}</span>
        {open ? <ChevronUp className="h-4 w-4 text-[#6b7280]" /> : <ChevronDown className="h-4 w-4 text-[#6b7280]" />}
      </button>
      {open && <div className="mt-3 space-y-3">{children}</div>}
    </div>
  );
}

function CheckboxRow({ checked, label, sublabel, onChange }) {
  return (
    <label className="flex cursor-pointer items-center gap-3 rounded-lg px-2 py-1.5 hover:bg-[#f8f9fb]">
      <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-md border ${checked ? "border-[#1c2260] bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820]" : "border-[#d1d5db] bg-white"} transition-colors`}>
        {checked && <Check className="h-3.5 w-3.5 text-white" strokeWidth={3} />}
      </span>
      {/* Hidden input to prevent browser scroll jumping to focus */}
      <input type="checkbox" checked={checked} onChange={onChange} className="hidden" />
      <span className="flex min-w-0 flex-1 items-center justify-between gap-3">
        <span className="truncate text-sm text-[#1a1d23]">{label}</span>
        {sublabel && <span className="shrink-0 text-xs text-[#9ca3af]">{sublabel}</span>}
      </span>
    </label>
  );
}

function RangePair({ min, max, value, onChange, format, step = 1 }) {
  const updateMin = (next) => onChange([Math.min(Number(next), value[1]), value[1]]);
  const updateMax = (next) => onChange([value[0], Math.max(Number(next), value[0])]);
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between rounded-lg border border-[#e2e4e9] bg-white px-3 py-2">
        <span className="font-['SF_Mono','Fira_Code','Consolas',monospace] text-xs font-semibold text-[#1c2260]">{format(value[0])}</span>
        <span className="text-xs text-[#9ca3af]">to</span>
        <span className="font-['SF_Mono','Fira_Code','Consolas',monospace] text-xs font-semibold text-[#1c2260]">{format(value[1])}</span>
      </div>
      <div className="space-y-2">
        <input type="range" min={min} max={max} step={step} value={value[0]} onChange={(e) => updateMin(e.target.value)} className="h-2 w-full appearance-none rounded-lg bg-[#e2e4e9] accent-[#1c2260]" />
        <input type="range" min={min} max={max} step={step} value={value[1]} onChange={(e) => updateMax(e.target.value)} className="h-2 w-full appearance-none rounded-lg bg-[#e2e4e9] accent-[#1c2260]" />
      </div>
    </div>
  );
}

function SelectField({ value, onChange, options, placeholder }) {
  return (
    <div className="relative min-w-0">
      <select value={value} onChange={(e) => onChange(e.target.value)} className="h-10 w-full appearance-none rounded-lg border border-[#e2e4e9] bg-white px-3 pr-9 text-sm text-[#1a1d23] outline-none transition focus:border-[#1c2260] focus:ring-2 focus:ring-[#1c2260]/10 truncate">
        <option value="">{placeholder}</option>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
      <ChevronDown className="pointer-events-none absolute right-3 top-3 h-4 w-4 text-[#6b7280]" />
    </div>
  );
}

function Stars({ rating }) {
  if (!rating) return <span className="text-xs text-[#9ca3af]">Not applicable</span>;
  return (
    <span className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((star) => <Star key={star} className={`h-3.5 w-3.5 ${star <= rating ? "fill-[#d4a843] text-[#d4a843]" : "fill-[#e5e7eb] text-[#e5e7eb]"}`} />)}
    </span>
  );
}

function WishlistButton({ school, isInList, onAdd, variant = 'full' }) {
  const isFull = school.slots_available === 0 && school.type !== 'public';
  if (variant === 'compact') {
    return (
      <button
        onClick={() => onAdd(school)}
        disabled={isFull || isInList}
        title={isFull ? 'No slots available' : 'Add to wishlist'}
        className={`shrink-0 p-2 rounded-lg transition ${isInList ? 'text-red-500 bg-red-50' : isFull ? 'text-slate-300 bg-slate-50 cursor-not-allowed' : 'text-[#1c2260] bg-blue-50 hover:bg-blue-100'}`}
      >
        {isInList ? <Heart className="fill-current h-4 w-4" /> : <Plus className="h-4 w-4" />}
      </button>
    );
  }
  return (
    <button
      onClick={() => onAdd(school)}
      disabled={isInList || isFull}
      className={`w-full h-11 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition ${isInList ? 'bg-red-50 text-red-600 border border-red-200' : isFull ? 'bg-slate-100 text-slate-400 cursor-not-allowed' : 'bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white hover:opacity-90'}`}
    >
      {isInList ? 'Added to Wishlist' : isFull ? 'No Slots Available' : 'Add to Wishlist'}
      {isInList ? <Heart className="fill-current h-4 w-4" /> : !isFull && <Plus className="h-4 w-4" />}
    </button>
  );
}

function ResultCard({ school, selected, onSelect, onAddToWishlist, isInWishlist }) {
  const isEscParticipant = school.esc_subsidy > 0;

  return (
    <div className={`w-full rounded-xl border bg-white p-4 text-left shadow-sm transition hover:-translate-y-0.5 ${selected ? "border-[#1c2260] ring-4 ring-[#1c2260]/10" : "border-[#e2e4e9]"}`}>
      <div className="flex items-start justify-between gap-3">
        <button type="button" onClick={() => onSelect(school)} className="min-w-0 text-left flex-1 focus:outline-none">
          <h3 className="line-clamp-2 text-sm font-semibold leading-snug text-[#1a1d23]">{school.name}</h3>
          <p className="mt-1 text-xs text-[#6b7280] truncate">{school.municipality}, {school.province}</p>
        </button>
        <WishlistButton school={school} isInList={isInWishlist} onAdd={onAddToWishlist} variant="compact" />
      </div>

      {/* UPDATED: Only rendering the School Type badge now */}
      <div className="mt-3 flex gap-2">
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${isEscParticipant ? "bg-[#16a34a]/10 text-[#16a34a] border border-[#16a34a]/20" : school.type === 'public' ? "bg-[#1c2260]/10 text-[#1c2260] border border-[#1c2260]/20" : "bg-[#f59e0b]/10 text-[#f59e0b] border border-[#f59e0b]/20"}`}>
          {isEscParticipant ? "Private with ESC" : school.type === 'public' ? "Public" : "Private No ESC"}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-[0.08em] text-[#9ca3af] truncate">Distance</p>
          <p className="mt-1 text-sm font-semibold text-[#1a1d23] truncate">{school.distance_km} km</p>
        </div>
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-[0.08em] text-[#9ca3af] truncate">Net Cost</p>
          <p className="mt-1 text-sm font-semibold text-[#1a1d23] truncate">{pesos(school.net_cost)}</p>
        </div>
        <div className="min-w-0">
          <p className="text-[11px] uppercase tracking-[0.08em] text-[#9ca3af] truncate">Slots</p>
          <p className="mt-1 text-sm font-semibold text-[#1a1d23] truncate">
            {isEscParticipant || school.type === 'public' ? school.slots_available : "—"}
          </p>
        </div>
      </div>

      {(isEscParticipant || school.type === 'public') && (
        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[#f0f1f4]">
          <div className={`h-full rounded-full ${slotTone(school)}`} style={{ width: `${pct(school.slots_available, school.slots_total)}%` }} />
        </div>
      )}
    </div>
  );
}

function SchoolInfoCard({ school, onClose, onAddToWishlist, isInWishlist }) {
  if (!school) return null;
  
  const isEscParticipant = school.esc_subsidy > 0;
  const meta = typeMeta[school.type];
  
  const availablePct = pct(school.slots_available, school.slots_total);
  const point = projectPoint(school);
  
  const xShift = point.x > (MAP_WIDTH / 2) ? "max(-105%, -1 * calc(100vw - 460px))" : "18px";
  const yShift = point.y > (MAP_HEIGHT / 2) ? "max(-85%, -1 * calc(100vh - 120px))" : "-18px";
  
  const cardPosition = {
    left: `${(point.x / MAP_WIDTH) * 100}%`,
    top: `${(point.y / MAP_HEIGHT) * 100}%`,
    transform: `translate(${xShift}, ${yShift})`,
  };

  return (
    <div
      className="absolute z-30 hidden w-[340px] max-h-[85vh] overflow-y-auto custom-scrollbar rounded-[20px] border border-[#e2e4e9] bg-white/95 p-5 shadow-[0_18px_48px_rgba(26,29,35,0.16),0_4px_12px_rgba(0,0,0,0.08)] backdrop-blur md:block transition-all duration-300"
      style={cardPosition}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded ${meta.badge}`}>
              {meta.label}
            </span>
          </div>
          <h3 className="mt-2 text-lg font-semibold leading-tight text-[#1a1d23]">{school.name}</h3>
        </div>
        <button type="button" onClick={onClose} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-[#e2e4e9] text-[#6b7280] transition hover:bg-[#f8f9fb] focus:outline-none">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 border-y border-[#e2e4e9] py-4">
        <div className="flex items-center gap-2"><MapPin className="h-4 w-4 text-[#1c2260]" /><span className="text-sm text-[#1a1d23]">{school.distance_km} km from you</span></div>
        <div className="flex items-center gap-2"><Clock3 className="h-4 w-4 text-[#1c2260]" /><span className="text-sm text-[#1a1d23]">~{school.commute_minutes} min</span></div>
      </div>

      <div className="mt-4 space-y-2">
        <div className="flex items-center justify-between text-sm"><span className="text-[#6b7280]">Tuition Fee</span><span className="font-semibold text-[#1a1d23]">{pesos(school.tuition)}</span></div>
        <div className="flex items-center justify-between text-sm">
            <span className="text-[#6b7280]">ESC Subsidy</span>
            <span className={`font-semibold ${isEscParticipant ? "text-[#16a34a]" : "text-[#9ca3af]"}`}>
                {isEscParticipant ? `-${pesos(school.esc_subsidy)}` : "None Available"}
            </span>
        </div>
        <div className="mt-2 flex items-center justify-between border-t border-[#e2e4e9] pt-3">
            <span className="text-sm font-semibold text-[#1a1d23]">Net Cost</span>
            <span className="text-xl font-bold text-[#1c2260]">{pesos(school.net_cost)}/yr</span>
        </div>
      </div>

      {isEscParticipant || school.type === 'public' ? (
        <div className="mt-5 rounded-xl bg-[#f8f9fb] p-4 border border-[#16a34a]/10">
          <span className="text-sm font-semibold text-[#1a1d23]">Available {isEscParticipant ? "ESC subsidy slots" : "slots"}: {school.slots_available}</span>
        </div>
      ) : (
        <div className="mt-5 rounded-xl bg-slate-50 p-4 border border-dashed border-slate-200">
            <p className="text-xs text-center text-slate-500 italic">This institution does not currently offer ESC subsidized slots.</p>
        </div>
      )}

      <div className="mt-4">
        <WishlistButton school={school} isInList={isInWishlist} onAdd={onAddToWishlist} />
      </div>
    </div>
  );
}

const MAP_STYLES = [
  { id: 'satellite', label: 'Satellite', url: 'mapbox://styles/mapbox/satellite-streets-v12' },
  { id: 'streets',   label: 'Streets',   url: 'mapbox://styles/mapbox/streets-v12' },
  { id: 'light',     label: 'Light',     url: 'mapbox://styles/mapbox/light-v11' },
];

const SCHOOL_GLOW_LAYER = {
  id: 'schools-glow',
  type: 'circle',
  paint: {
    'circle-radius': 22,
    'circle-color': [
      'match', ['get', 'type'],
      'public',         '#60a5fa',
      'private_esc',    '#4ade80',
      'private_no_esc', '#fbbf24',
      '#94a3b8'
    ],
    'circle-opacity': 0.35,
    'circle-blur': 1,
  },
};

const SCHOOL_DOT_LAYER = {
  id: 'schools',
  type: 'circle',
  paint: {
    'circle-radius': 7,
    'circle-color': [
      'match', ['get', 'type'],
      'public',         '#60a5fa',
      'private_esc',    '#4ade80',
      'private_no_esc', '#fbbf24',
      '#94a3b8'
    ],
    'circle-stroke-width': 2,
    'circle-stroke-color': 'rgba(255,255,255,0.9)',
    'circle-opacity': 1,
  },
};

function PhilippinesMap({ filteredSchools, selectedSchool, hoveredId, onHover, onSelect, comingSoon }) {
  const [styleId, setStyleId] = useState('satellite');
  const [hoverInfo, setHoverInfo] = useState(null);

  // Selected school is excluded from GeoJSON and rendered as a pulsing Marker instead
  const geojson = useMemo(() => ({
    type: 'FeatureCollection',
    features: filteredSchools
      .filter(s => s.id !== selectedSchool?.id)
      .map(s => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [s.lng, s.lat] },
        properties: { id: s.id, type: s.type, name: s.name },
      })),
  }), [filteredSchools, selectedSchool?.id]);

  const handleClick = useCallback((e) => {
    const feature = e.features?.[0];
    if (!feature) return;
    const school = filteredSchools.find(s => s.id === feature.properties.id);
    if (school) onSelect(school);
  }, [filteredSchools, onSelect]);

  const handleMouseMove = useCallback((e) => {
    const feature = e.features?.[0];
    if (feature) {
      onHover(feature.properties.id);
      setHoverInfo({ name: feature.properties.name, type: feature.properties.type, x: e.point.x, y: e.point.y });
      e.target.getCanvas().style.cursor = 'pointer';
    } else {
      onHover(null);
      setHoverInfo(null);
      e.target.getCanvas().style.cursor = '';
    }
  }, [onHover]);

  const activeStyle = MAP_STYLES.find(s => s.id === styleId);

  return (
    <div className="relative h-full w-full">
      <Map
        initialViewState={{ longitude: 121.0, latitude: 14.65, zoom: 9 }}
        style={{ width: '100%', height: '100%' }}
        mapStyle={activeStyle.url}
        mapboxAccessToken={import.meta.env.VITE_MAPBOX_TOKEN}
        logoPosition="bottom-right"
        interactiveLayerIds={['schools']}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
      >
        <NavigationControl position="top-right" showCompass={false} />
        <Source id="schools-source" type="geojson" data={geojson}>
          <Layer {...SCHOOL_GLOW_LAYER} />
          <Layer {...SCHOOL_DOT_LAYER} />
        </Source>

        {selectedSchool && (
          <Marker longitude={selectedSchool.lng} latitude={selectedSchool.lat} anchor="center">
            <div className="relative flex h-8 w-8 items-center justify-center">
              <span
                className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-25"
                style={{ backgroundColor: typeMeta[selectedSchool.type].dot }}
              />
              <span
                className="relative inline-flex h-4 w-4 rounded-full border-2 border-white shadow-lg"
                style={{ backgroundColor: typeMeta[selectedSchool.type].dot }}
              />
            </div>
          </Marker>
        )}
      </Map>

      {/* Hover tooltip */}
      {hoverInfo && (
        <div
          className="pointer-events-none absolute z-20 rounded-xl border border-slate-100 bg-white/95 px-3 py-2.5 shadow-lg backdrop-blur-sm"
          style={{ left: hoverInfo.x + 14, top: hoverInfo.y - 48 }}
        >
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 flex-shrink-0 rounded-full" style={{ backgroundColor: typeMeta[hoverInfo.type]?.dot }} />
            <span className="max-w-[200px] truncate text-xs font-semibold text-[#1a1d23]">{hoverInfo.name}</span>
          </div>
          <p className="mt-0.5 pl-4 text-[10px] font-medium uppercase tracking-wide text-slate-400">
            {typeMeta[hoverInfo.type]?.label}
          </p>
        </div>
      )}

      {/* Legend */}
      {/* Map style switcher */}
      <div className="absolute bottom-14 right-5 z-10 flex overflow-hidden rounded-xl border border-slate-100 bg-white/90 shadow-lg backdrop-blur-sm">
        {MAP_STYLES.map((s, i) => (
          <button
            key={s.id}
            type="button"
            onClick={() => setStyleId(s.id)}
            className={[
              'px-3 py-2 text-[11px] font-semibold transition-colors',
              i > 0 ? 'border-l border-slate-100' : '',
              styleId === s.id ? 'bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white' : 'text-slate-500 hover:bg-slate-50',
            ].join(' ')}
          >
            {s.label}
          </button>
        ))}
      </div>

      {comingSoon && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-[#f8f9fb]/82 p-8 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-[20px] border border-[#e2e4e9] bg-white p-7 text-center shadow-[0_18px_48px_rgba(26,29,35,0.16),0_4px_12px_rgba(0,0,0,0.08)]">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-[#1c2260]/10">
              <Sparkles className="h-6 w-6 text-[#1c2260]" />
            </div>
            <h2 className="mt-4 text-xl font-semibold text-[#1a1d23]">{comingSoon} is coming soon</h2>
            <p className="mt-2 text-sm leading-6 text-[#6b7280]">
              This live demo focuses on the family-facing Student View. The same access and capacity data model can power school operations and DepEd policy simulations.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function getDocList(category, answers) {
  const incomeDoc = {
    local: 'Income Tax Return (ITR), Certificate of Employment, or recent Payslip',
    abroad: 'Certificate of Employment, Employment Contract, or recent Payslip',
    business: 'Notarized Affidavit (business income)',
    unemployed: 'Certificate of Tax Exemption or Barangay Certificate of Indigency',
  };
  const segDoc = {
    '4ps': 'Copy of 4Ps ID (DSWD)',
    gidca: 'Barangay Certification of GIDCA residency',
    ip: 'Certificate of Indigenous People Membership — NCIP',
    pwd: 'PWD ID issued by LGU',
    special: 'Medical or psychological assessment',
    cbms: 'CBMS poverty assessment document',
  };
  const base = [
    'Valid ID (National ID, Birth Certificate, or Passport)',
    'Accomplished ESC Application Form (Annex D)',
  ];
  const affidavit = `Affidavit of Family's Financial Capacity (Annex F) — ${incomeDoc[answers.employment] || 'income proof document'}`;
  if (category === 'A') {
    const extra = (answers.segs || []).filter(s => s !== 'none' && segDoc[s]).map(s => segDoc[s]);
    return [...base, "SF9 — Learner's Progress Report Card", ...extra];
  }
  if (category === 'B') return [...base, "SF9 — Learner's Progress Report Card", affidavit];
  if (category === 'C') return [...base, 'Certificate of Rating from BEA (ALS A&E / PEPT)', affidavit];
  if (category === 'D') return [...base, "SF9 — Learner's Progress Report Card", affidavit];
  return [];
}

const STORAGE_KEY = 'paaral_v3_account';
const TEST_EMAIL = '100000000001@deped.gov.ph';
const TEST_LRN = '100000000001';
const LEARNER_RECORD = {
  firstName: 'Juan', mi: 'M', lastName: 'dela Cruz',
  school: 'Bagumbayan Elementary School', grade: 'Grade 6',
  municipality: 'Taguig City, Metro Manila',
  division: 'Division of Taguig-Pateros',
};
const catMeta = {
  A: { label: 'Category A — Social Equity Group', tw: 'bg-[#f5f3ff] border-[#c4b5fd] text-[#5b21b6]', desc: 'Highest priority. Applies to learners from equity-protected groups regardless of income.' },
  B: { label: 'Category B — Public School Graduate', tw: 'bg-[#eff6ff] border-[#bfdbfe] text-[#1d4ed8]', desc: 'Public school graduate with poor to middle-class household income.' },
  C: { label: 'Category C — ALS / PEPT Passer', tw: 'bg-[#f0fdfa] border-[#99f6e4] text-[#0f766e]', desc: 'ALS A&E Test or PEPT passer with eligible income.' },
  D: { label: 'Category D — Private School Graduate', tw: 'bg-[#fffbeb] border-[#fde68a] text-[#b45309]', desc: 'Private school graduate with poor to middle-class household income.' },
};

// ── APPLICATION STATE MACHINE ─────────────────────────────────────────
// Decoupled model: PAARAL tracks the ESC application track only. School
// admission/enrollment is an independent, unmodeled track — 'granted' means
// the ESC certificate is secured, full stop, regardless of enrollment timing.
const POST_SUBMISSION_STATES = new Set(['submitted', 'granted', 'non_esc']);

const VALID_TRANSITIONS = {
  eligibility:  ['submitted'],
  not_eligible: ['non_esc'],
  submitted:    ['granted', 'non_esc', 'eligibility'], // 'eligibility' = stop, choose different schools
  granted:      [],
  non_esc:      [],
};

// Per-school ESC status — private schools only. Public schools are never
// entered into the ESC pursuit; they're just the hasPublicAlternative
// guaranteed-placement checkbox. 'granted'/'rejected' are both terminal at
// the school level — no admission dependency either way.
const ESC_SCHOOL_TRANSITIONS = {
  submitted:      ['granted', 'rejected', 'docs_pending'],
  docs_pending:   ['docs_submitted'],
  docs_submitted: ['granted', 'rejected'],
  granted:        [],
  rejected:       [],
};

const REJECTED_STATES = new Set(['rejected']);

function useApplicationState({ account, updateAccount, wishlist, hasPrivateChoice, hasPublicAlternative, docsReady, generalSurveyComplete, escSurveyComplete }) {
  const applicationState = account?.applicationState ?? 'eligibility';
  const isPostSubmission = POST_SUBMISSION_STATES.has(applicationState);
  const canSubmitEsc = applicationState === 'eligibility'
    && hasPrivateChoice
    && hasPublicAlternative
    && docsReady
    && generalSurveyComplete
    && escSurveyComplete;
  const canEnrollNonEsc = applicationState === 'not_eligible'
    && wishlist.length > 0
    && generalSurveyComplete;

  const advance = (toState, extra = {}) => {
    const valid = VALID_TRANSITIONS[applicationState] ?? [];
    if (!valid.includes(toState)) return;
    updateAccount({ applicationState: toState, ...extra });
  };

  return { applicationState, isPostSubmission, canSubmitEsc, canEnrollNonEsc, advance };
}

export default function PAARALStudentMockup() {
  // ── ACCOUNT ──────────────────────────────────────────────────
  const [account, setAccount] = useState(null);
  const updateAccount = (patch) => {
    setAccount(prev => {
      const next = { ...prev, ...patch };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };
  // Mockup-only: logout wipes local account + questionnaire state so each demo
  // login starts clean. In production, account state is server-persisted and
  // survives logout — this reset should not carry over.
  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    setAccount(null);
    setAppView('hero');
    setDrawerOpen(false);
    setEligStep('schoolType');
    setEligHistory([]);
    setEligAnswers({ escIntent: true, schoolType: null, segs: [], income: null, employment: null });
  };

  // ── APP VIEW ──────────────────────────────────────────────────
  const [appView, setAppView] = useState('hero');
  const [browseTab, setBrowseTab] = useState('map');
  const [leftPanel, setLeftPanel] = useState(null);
  const [leftPanelTab, setLeftPanelTab] = useState('about');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState('choices');

  // ── LOGIN MODAL ───────────────────────────────────────────────
  const [showLogin, setShowLogin] = useState(false);
  const [loginEmail, setLoginEmail] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState('');
  const [loginConfirmed, setLoginConfirmed] = useState(false);

  // ── ELIGIBILITY QUESTIONNAIRE ─────────────────────────────────
  const [eligStep, setEligStep] = useState('schoolType');
  const [eligHistory, setEligHistory] = useState([]);
  const [eligAnswers, setEligAnswers] = useState({ escIntent: true, schoolType: null, segs: [], income: null, employment: null });

  // ── FILTERS ───────────────────────────────────────────────────
  const [searchTerm, setSearchTerm] = useState('');
  const [region, setRegion] = useState('');
  const [province, setProvince] = useState('');
  const [municipality, setMunicipality] = useState('');
  const [barangay, setBarangay] = useState('');
  const [distance, setDistance] = useState([0, 100]);
  const [tuition, setTuition] = useState([0, 250000]);
  const [commuteBuckets, setCommuteBuckets] = useState([]);
  const [schoolTypes, setSchoolTypes] = useState([]);
  const [openSections, setOpenSections] = useState(accordionDefaults);
  const [filtersCollapsed, setFiltersCollapsed] = useState(false);
  const [resultsCollapsed, setResultsCollapsed] = useState(false);
  const [selectedSchool, setSelectedSchool] = useState(null);
  const [hoveredId, setHoveredId] = useState(null);
  const [profileSection, setProfileSection] = useState('overview');
  const profileScrollRef = useRef(null);
  const overviewRef = useRef(null);
  const characteristicsRef = useRef(null);
  const feeRef = useRef(null);
  const resultsScrollRef = useRef(null);
  const selectedCardRef = useRef(null);

  const [surveyAnswers, setSurveyAnswers] = useState({ ease: null, helpful: null, concern: null, suggestions: '' });

  // ── DERIVED ───────────────────────────────────────────────────
  const wishlist = useMemo(() => {
    if (!account?.wishlistIds) return [];
    return account.wishlistIds.map(id => schools.find(s => s.id === id)).filter(Boolean);
  }, [account?.wishlistIds]);

  const filteredSchools = useMemo(() => {
    return schools.filter(s => {
      if (searchTerm && !s.name.toLowerCase().includes(searchTerm.toLowerCase()) && !s.municipality.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      if (region && s.region !== region) return false;
      if (province && s.province !== province) return false;
      if (municipality && s.municipality !== municipality) return false;
      if (barangay && s.barangay !== barangay) return false;
      if (s.distance_km < distance[0] || s.distance_km > distance[1]) return false;
      if (s.net_cost < tuition[0] || s.net_cost > tuition[1]) return false;
      if (!commuteBucketMatches(s, commuteBuckets)) return false;
      if (!schoolTypeMatches(s, schoolTypes)) return false;
      return true;
    });
  }, [searchTerm, region, province, municipality, barangay, distance, tuition, commuteBuckets, schoolTypes]);

  const uploadedDocs = account?.uploadedDocs || [];
  const hasPublicAlternative = wishlist.some(s => s.type === 'public');
  const requiredDocs = account?.category ? getDocList(account.category, account.eligAnswers || {}) : [];
  const docsReady = requiredDocs.length > 0 && requiredDocs.every(d => uploadedDocs.includes(d));
  const generalSurveyComplete = Boolean(surveyAnswers.ease && surveyAnswers.helpful);
  const escSurveyComplete = Boolean(surveyAnswers.concern);
  const isInWishlist = (school) => (account?.wishlistIds || []).includes(school.id);

  // ── PER-SCHOOL ESC STATUS (private schools only — public is out of scope,
  // it's just the guaranteed-placement safety net, never entered into the pursuit) ──
  const escStatuses = account?.escStatuses || {};
  const privateChoices = wishlist.filter(s => s.type !== 'public');
  const hasPrivateChoice = privateChoices.length > 0;
  const activeChoice = privateChoices.find(s => ['submitted', 'docs_pending', 'docs_submitted'].includes(escStatuses[s.id]));
  const lastEngagedIndex = privateChoices.reduce((last, s, i) => (escStatuses[s.id] ? i : last), -1);
  const lastEngagedChoice = lastEngagedIndex >= 0 ? privateChoices[lastEngagedIndex] : null;
  const nextChoice = privateChoices[lastEngagedIndex + 1] || null;
  const grantedChoice = privateChoices.find(s => escStatuses[s.id] === 'granted');

  const { applicationState: appState, isPostSubmission, canSubmitEsc, canEnrollNonEsc, advance } = useApplicationState({
    account, updateAccount, wishlist, hasPrivateChoice, hasPublicAlternative, docsReady, generalSurveyComplete, escSurveyComplete,
  });

  // ── PROFILE SCROLL SYNC ───────────────────────────────────────
  useEffect(() => {
    const container = profileScrollRef.current;
    if (!container) return;
    const handleScroll = () => {
      const sects = [
        { id: 'overview', ref: overviewRef },
        { id: 'characteristics', ref: characteristicsRef },
        { id: 'fee', ref: feeRef },
      ];
      for (const { id, ref } of sects.slice().reverse()) {
        if (ref.current && ref.current.offsetTop <= container.scrollTop + 80) {
          setProfileSection(id);
          break;
        }
      }
    };
    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [leftPanel, selectedSchool]);

  // ── HELPERS ───────────────────────────────────────────────────
  const toggleSection = (id) => setOpenSections(o => ({ ...o, [id]: !o[id] }));

  const computeCategory = (answers) => {
    const { schoolType, segs, income } = answers;
    const hasSegs = segs && segs.some(s => s !== 'none');
    if (schoolType === 'public' && hasSegs) return 'A';
    if (income === 'above') return null;
    if (schoolType === 'public') return 'B';
    if (schoolType === 'als') return 'C';
    if (schoolType === 'private') return 'D';
    return null;
  };

  const eligBack = () => {
    if (eligHistory.length === 0) return;
    const prev = eligHistory[eligHistory.length - 1];
    setEligStep(prev.step);
    setEligAnswers(prev.answers);
    setEligHistory(h => h.slice(0, -1));
  };

  const eligGo = (step, patch = {}) => {
    setEligHistory(h => [...h, { step: eligStep, answers: { ...eligAnswers } }]);
    setEligAnswers(a => ({ ...a, ...patch }));
    setEligStep(step);
  };

  const nextEligStep = (step, answers) => {
    if (step === 'schoolType') return answers.schoolType === 'public' ? 'seg' : 'income';
    if (step === 'seg') return answers.segs.some(s => s !== 'none') ? 'result' : 'income';
    if (step === 'income') return answers.income === 'above' ? 'result' : 'employment';
    return 'result';
  };

  const completeEligibility = () => {
    const category = computeCategory(eligAnswers);
    // 'eligibility' still covers browsing/wishlist-building once a category is assigned —
    // there's no separate stored state for that sub-phase (see renderStateBadge).
    updateAccount({ category, eligAnswers, applicationState: category ? 'eligibility' : 'not_eligible' });
    setAppView('browse');
  };

  const addToWishlist = (school) => {
    if (!account) { setShowLogin(true); return; }
    if (isPostSubmission) return;
    const ids = account.wishlistIds || [];
    if (ids.includes(school.id)) return;
    updateAccount({ wishlistIds: [...ids, school.id] });
  };

  const removeFromWishlist = (schoolId) => {
    if (!account || isPostSubmission) return;
    updateAccount({ wishlistIds: (account.wishlistIds || []).filter(id => id !== schoolId) });
  };

  const handleSubmitEsc = () => {
    if (!canSubmitEsc) return;
    const rank1 = privateChoices[0];
    updateAccount({
      escStatuses: { ...escStatuses, [rank1.id]: 'submitted' },
      surveyAnswers, uploadedDocs,
    });
    advance('submitted');
    setDrawerTab('status');
  };

  const handleEnrollNonEsc = () => {
    if (!canEnrollNonEsc) return;
    const school = wishlist[0];
    advance('non_esc', { nonEscSchoolId: school.id, surveyAnswers });
    setDrawerTab('status');
  };

  // Advance one specific PRIVATE school's ESC status. Only reaching 'granted'
  // ends the account-level pursuit — 'rejected' re-opens the next-rank prompt.
  const advanceSchool = (schoolId, toState) => {
    const current = escStatuses[schoolId];
    const valid = ESC_SCHOOL_TRANSITIONS[current] ?? [];
    if (!valid.includes(toState)) return;
    const nextEscStatuses = { ...escStatuses, [schoolId]: toState };
    if (toState === 'granted') {
      advance('granted', { escStatuses: nextEscStatuses });
    } else {
      updateAccount({ escStatuses: nextEscStatuses });
    }
  };

  // After a rejection, apply to the next-ranked PRIVATE choice — explicit
  // opt-in, never automatic.
  const applyToNextRank = () => {
    if (!nextChoice) return;
    updateAccount({ escStatuses: { ...escStatuses, [nextChoice.id]: 'submitted' } });
  };

  const continueWithoutSubsidy = (schoolId) => {
    advance('non_esc', { nonEscSchoolId: schoolId });
    setDrawerTab('status');
  };

  const applyAgainDifferentSchool = () => {
    advance('eligibility', { wishlistIds: [], escStatuses: {} });
    setDrawerTab('choices');
  };

  const closeLogin = () => {
    setShowLogin(false);
    setLoginConfirmed(false);
    setLoginEmail('');
    setLoginError('');
  };

  const handleEmailSubmit = () => {
    setLoginError('');
    if (loginEmail.trim().toLowerCase() !== TEST_EMAIL.toLowerCase()) {
      setLoginError('Email not found. Use 100000000001@deped.gov.ph for this demo.');
      return;
    }
    setLoginLoading(true);
    setTimeout(() => { setLoginLoading(false); setLoginConfirmed(true); }, 800);
  };

  const handleCreateAccount = () => {
    const newAccount = {
      email: TEST_EMAIL, lrn: TEST_LRN,
      name: `${LEARNER_RECORD.firstName} ${LEARNER_RECORD.mi}. ${LEARNER_RECORD.lastName}`,
      category: null, eligAnswers: null,
      applicationState: 'eligibility',
      wishlistIds: [],
      escStatuses: {},
      surveyAnswers: { ease: null, helpful: null, concern: null, suggestions: '' },
      uploadedDocs: [],
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newAccount));
    setAccount(newAccount);
    closeLogin();
    setAppView('eligibility');
  };


  const handleMapSelect = (school) => {
    setSelectedSchool(school);
    setLeftPanel('school');
  };

  const scrollToSection = (ref) => {
    if (profileScrollRef.current && ref.current) {
      profileScrollRef.current.scrollTo({ top: ref.current.offsetTop, behavior: 'smooth' });
    }
  };

  const userLocation = { lat: 14.5195, lng: 121.0540 };

  const renderStateBadge = (state) => {
    // Pre-submission (eligibility/not_eligible) shows no status badge at all —
    // there's no application yet.
    if (!POST_SUBMISSION_STATES.has(state)) return null;
    const map = {
      submitted:   ['bg-blue-100 text-blue-800 border-blue-300', 'ESC Application In Progress'],
      granted:     ['bg-purple-50 text-purple-700 border-purple-200', 'ESC Certificate Granted'],
      non_esc:     ['bg-slate-100 text-slate-600 border-slate-300', 'Non-ESC Pathway'],
    };
    const [tw, label] = map[state] || ['bg-slate-100 text-slate-500 border-slate-200', state];
    return <span className={`inline-block text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${tw}`}>{label}</span>;
  };

  // ── DRAWER TABS ───────────────────────────────────────────────
  const drawerTabList = isPostSubmission ? ['status', 'documents', 'choices'] : ['choices', 'documents', 'survey'];

  // ── STATUS TAB CONTENT (per-school — the ESC track only; admission/enrollment
  // is an independent, unmodeled track per the decoupled "Portable Eligibility" model) ──
  const schoolStatusConfigs = {
    submitted: {
      icon: <Clock3 className="h-8 w-8 text-blue-500" />,
      title: 'ESC Application Submitted',
      desc: name => `Your ESC application to ${name} has been received. You will be notified once it has been reviewed.`,
      color: 'bg-blue-50 border-blue-200',
      demo: [
        { label: 'Simulate: Granted', next: 'granted', style: 'bg-green-600 text-white' },
        { label: 'Simulate: Rejected', next: 'rejected', style: 'bg-red-600 text-white' },
        { label: 'Simulate: Additional Doc Requested', next: 'docs_pending', style: 'bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white' },
      ],
    },
    rejected: {
      icon: <AlertCircle className="h-8 w-8 text-red-500" />,
      title: 'ESC Application Not Approved',
      desc: name => `Your ESC application to ${name} was not approved this cycle.`,
      color: 'bg-red-50 border-red-200',
      demo: [],
    },
    docs_pending: {
      icon: <FileCheck className="h-8 w-8 text-amber-500" />,
      title: 'Additional Document Requested',
      desc: name => `${name}'s ESC School Committee has requested an additional document. Please check the Documents tab.`,
      color: 'bg-amber-50 border-amber-200',
      demo: [],
    },
    docs_submitted: {
      icon: <FileCheck className="h-8 w-8 text-blue-500" />,
      title: 'Additional Document Under Review',
      desc: name => `Your documents for ${name} have been submitted and are being reviewed by the ESC School Committee.`,
      color: 'bg-blue-50 border-blue-200',
      demo: [
        { label: 'Simulate: Granted', next: 'granted', style: 'bg-green-600 text-white' },
        { label: 'Simulate: Rejected', next: 'rejected', style: 'bg-red-600 text-white' },
      ],
    },
    granted: {
      icon: <Award className="h-8 w-8 text-purple-500" />,
      title: 'ESC Certificate Granted 🎉',
      desc: name => `Your ESC subsidy for ${name} has been confirmed.`,
      color: 'bg-purple-50 border-purple-200',
      demo: [],
    },
  };

  // ═══════════════════════════════════════════════════════════════
  // LOGIN MODAL
  // ═══════════════════════════════════════════════════════════════
  const loginModal = showLogin && (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] opacity-90" onClick={closeLogin} />
      <div className="relative z-10 w-full max-w-md mx-4 bg-white rounded-[22px] p-8 shadow-2xl">
        <div className="flex items-center justify-between mb-6">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">DepEd ICTS Portal</p>
            <h2 className="text-xl font-bold text-[#1a1d23] mt-0.5">Sign in with your DepEd email</h2>
          </div>
          <button onClick={closeLogin} className="h-9 w-9 flex items-center justify-center rounded-full border border-slate-200 text-slate-400 hover:bg-slate-50">
            <X className="h-4 w-4" />
          </button>
        </div>
        {!loginConfirmed ? (
          <>
            <p className="text-sm text-slate-500 mb-5">Your DepEd email follows the format: <span className="font-mono text-slate-700">LRN@deped.gov.ph</span></p>
            <input
              type="email"
              value={loginEmail}
              onChange={e => { setLoginEmail(e.target.value); setLoginError(''); }}
              onKeyDown={e => e.key === 'Enter' && handleEmailSubmit()}
              placeholder="e.g. 100000000001@deped.gov.ph"
              className="w-full h-11 px-4 rounded-xl border border-slate-200 text-sm outline-none focus:border-[#1c2260] focus:ring-2 focus:ring-[#1c2260]/10"
            />
            {loginError && <p className="mt-2 text-xs text-red-600">{loginError}</p>}
            <button
              onClick={handleEmailSubmit}
              disabled={loginLoading || !loginEmail}
              className="mt-4 w-full h-11 rounded-xl bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white font-semibold text-sm disabled:opacity-50"
            >
              {loginLoading ? 'Verifying…' : 'Continue'}
            </button>
            <p className="mt-4 text-xs text-center text-slate-400">Demo: 100000000001@deped.gov.ph</p>
          </>
        ) : (
          <>
            <div className="rounded-xl bg-green-50 border border-green-200 p-4 mb-5">
              <p className="text-[10px] font-bold uppercase tracking-widest text-green-600 mb-2">Learner found in LIS</p>
              <p className="text-base font-bold text-slate-800">{LEARNER_RECORD.firstName} {LEARNER_RECORD.mi}. {LEARNER_RECORD.lastName}</p>
              <p className="text-xs text-slate-500 mt-1">{LEARNER_RECORD.school} · {LEARNER_RECORD.grade}</p>
              <p className="text-xs text-slate-500">{LEARNER_RECORD.municipality} · {LEARNER_RECORD.division}</p>
              <p className="text-xs text-slate-400 mt-1 font-mono">LRN: {TEST_LRN}</p>
            </div>
            <p className="text-sm text-slate-600 mb-5">Creating your PAARAL account starts your ESC eligibility assessment.</p>
            <button onClick={handleCreateAccount} className="w-full h-11 rounded-xl bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white font-semibold text-sm">
              Create My Account & Start
            </button>
            <button onClick={() => { setLoginConfirmed(false); setLoginEmail(''); }} className="mt-2 w-full h-9 text-xs text-slate-400 hover:text-slate-600">
              ← Use a different email
            </button>
          </>
        )}
      </div>
    </div>
  );

  // ═══════════════════════════════════════════════════════════════
  // HERO VIEW
  // ═══════════════════════════════════════════════════════════════
  if (appView === 'hero') {
    return (
      <div className="fixed inset-0 flex flex-col bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820]">
        <link href="https://fonts.googleapis.com/css2?family=Baskervville&display=swap" rel="stylesheet" />
        {loginModal}
        <div className="flex flex-col items-center justify-center flex-1 text-center px-6">
          <div className="flex items-center gap-4 mb-8">
            <img src="/assets/deped-logo.png" alt="DepEd" className="h-12" />
            <img src="/assets/ecair-logo.png" alt="ECAIR" className="h-8" />
          </div>
          <div className="font-['Baskervville',serif] text-[clamp(2.5rem,8vw,5.5rem)] font-normal tracking-[0.08em] text-white leading-none mb-3">
            PROJECT PAARAL
          </div>
          <p className="text-white/60 text-sm tracking-widest uppercase mb-2">Educational Service Contracting</p>
          <span className="px-3 py-0.5 rounded-full bg-white/10 border border-white/20 text-white/70 text-[10px] font-bold uppercase tracking-widest mb-12">BETA</span>
          <p className="text-white/70 max-w-md text-base leading-relaxed mb-10">
            Find and apply to Grade 7 ESC-partner schools near you. Browse without an account, or log in to start your ESC application.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 w-full max-w-xs">
            <button onClick={() => setAppView('browse')} className="flex-1 h-12 rounded-xl bg-white text-[#1c2260] font-bold text-sm hover:bg-white/90 transition">
              Browse Schools
            </button>
            <button
              onClick={() => {
                if (account) {
                  setAppView(account.applicationState === 'eligibility' ? 'eligibility' : 'browse');
                } else {
                  setShowLogin(true);
                }
              }}
              className="flex-1 h-12 rounded-xl border border-white/30 text-white font-semibold text-sm hover:bg-white/10 transition"
            >
              {account ? 'My Account →' : 'Log In'}
            </button>
          </div>
          {account && (
            <p className="mt-5 text-white/40 text-xs">
              Signed in as {account.name} ·{' '}
              <button onClick={logout} className="underline hover:text-white/70">Log out</button>
            </p>
          )}
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════
  // ELIGIBILITY VIEW
  // ═══════════════════════════════════════════════════════════════
  if (appView === 'eligibility') {
    const catResult = eligStep === 'result' ? computeCategory(eligAnswers) : null;
    const docList = catResult ? getDocList(catResult, eligAnswers) : [];
    const stepOrder = ['schoolType', 'seg', 'income', 'employment', 'result'];

    return (
      <div className="fixed inset-0 flex flex-col bg-[#f8f9fa]">
        <link href="https://fonts.googleapis.com/css2?family=Baskervville&display=swap" rel="stylesheet" />
        <div className="h-14 shrink-0 flex items-center justify-between px-6 bg-gradient-to-r from-[#1c2260] to-[#5a2d68] shadow z-10">
          <span className="font-['Baskervville',serif] text-white text-lg tracking-widest">PROJECT PAARAL</span>
          <div className="flex items-center gap-3">
            <span className="text-white/60 text-xs hidden sm:block">{account?.name}</span>
            <button onClick={logout} className="text-white/50 hover:text-white text-xs flex items-center gap-1.5">
              <LogOut className="h-3.5 w-3.5" /> Log out
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto flex items-start justify-center py-12 px-4">
          <div className="w-full max-w-lg bg-white rounded-[22px] border border-slate-200 shadow-lg p-8">
            <div className="flex gap-2 mb-6">
              {stepOrder.map(s => (
                <div key={s} className={`h-1.5 flex-1 rounded-full transition-colors ${
                  s === eligStep ? 'bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820]' :
                  eligHistory.some(h => h.step === s) ? 'bg-[#1c2260]/40' : 'bg-slate-200'
                }`} />
              ))}
            </div>

            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">ESC Eligibility Assessment</p>

            {eligStep === 'schoolType' && (
              <>
                <h2 className="text-xl font-bold text-slate-800 mb-2">How did you complete Grade 6?</h2>
                <p className="text-sm text-slate-500 mb-6">This determines your ESC category under E-GASTPE 2026 guidelines.</p>
                <div className="space-y-3">
                  {[
                    { v: 'public', label: 'Public elementary school', sub: 'Any DepEd-operated school' },
                    { v: 'private', label: 'Private elementary school', sub: 'Non-DepEd institution' },
                    { v: 'als', label: 'ALS or PEPT', sub: 'Alternative Learning System or Philippine Educational Placement Test' },
                  ].map(opt => (
                    <button key={opt.v} onClick={() => eligGo(nextEligStep('schoolType', { ...eligAnswers, schoolType: opt.v }), { schoolType: opt.v })}
                      className="w-full p-4 rounded-xl border border-slate-200 hover:border-[#1c2260] hover:bg-blue-50 text-left transition group">
                      <p className="font-semibold text-slate-800 text-sm group-hover:text-[#1c2260]">{opt.label}</p>
                      <p className="text-xs text-slate-400 mt-0.5">{opt.sub}</p>
                    </button>
                  ))}
                </div>
              </>
            )}

            {eligStep === 'seg' && (
              <>
                <h2 className="text-xl font-bold text-slate-800 mb-2">Do you belong to a Social Equity Group?</h2>
                <p className="text-sm text-slate-500 mb-5">Select all that apply.</p>
                <div className="space-y-2 mb-6">
                  {[
                    { v: '4ps', label: '4Ps / Pantawid Pamilyang Pilipino Program' },
                    { v: 'gidca', label: 'Geographically Isolated and Disadvantaged Community (GIDCA)' },
                    { v: 'ip', label: 'Indigenous People (IP)' },
                    { v: 'pwd', label: 'Person with Disability (PWD)' },
                    { v: 'special', label: 'Child with Special Needs' },
                    { v: 'cbms', label: 'CBMS-identified poor or near-poor household' },
                    { v: 'none', label: 'None of the above' },
                  ].map(opt => {
                    const checked = eligAnswers.segs.includes(opt.v);
                    return (
                      <button key={opt.v}
                        onClick={() => {
                          let segs = eligAnswers.segs;
                          if (opt.v === 'none') {
                            segs = checked ? [] : ['none'];
                          } else {
                            segs = checked
                              ? segs.filter(s => s !== opt.v)
                              : [...segs.filter(s => s !== 'none'), opt.v];
                          }
                          setEligAnswers(a => ({ ...a, segs }));
                        }}
                        className={`w-full p-3 rounded-xl border text-left text-sm flex items-center gap-3 transition ${checked ? 'border-[#1c2260] bg-blue-50 text-[#1c2260] font-medium' : 'border-slate-200 text-slate-700 hover:border-slate-300'}`}
                      >
                        <span className={`h-5 w-5 shrink-0 rounded-md border flex items-center justify-center ${checked ? 'bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] border-[#1c2260]' : 'border-slate-300'}`}>
                          {checked && <Check className="h-3.5 w-3.5 text-white" strokeWidth={3} />}
                        </span>
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
                <div className="flex gap-2">
                  <button onClick={eligBack} className="h-11 px-4 rounded-xl border border-slate-200 text-sm text-slate-600 hover:bg-slate-50">← Back</button>
                  <button
                    onClick={() => eligGo(nextEligStep('seg', eligAnswers))}
                    disabled={eligAnswers.segs.length === 0}
                    className="flex-1 h-11 rounded-xl bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white font-semibold text-sm disabled:opacity-40"
                  >
                    Continue →
                  </button>
                </div>
              </>
            )}

            {eligStep === 'income' && (
              <>
                <h2 className="text-xl font-bold text-slate-800 mb-2">Monthly household income</h2>
                <p className="text-sm text-slate-500 mb-6">PIDS income classification. Include all household members' income.</p>
                <div className="space-y-3 mb-6">
                  {[
                    { v: 'poor', label: 'Poor', sub: 'Less than ₱10,957/month' },
                    { v: 'low', label: 'Low income', sub: '₱10,957 – ₱21,194/month' },
                    { v: 'lower_middle', label: 'Lower middle class', sub: '₱21,194 – ₱43,828/month' },
                    { v: 'middle', label: 'Middle class', sub: '₱43,828 – ₱76,669/month' },
                    { v: 'above', label: 'Upper middle income or above', sub: 'More than ₱76,669/month — not eligible for ESC B/C/D' },
                  ].map(opt => (
                    <button key={opt.v} onClick={() => eligGo(nextEligStep('income', { ...eligAnswers, income: opt.v }), { income: opt.v })}
                      className="w-full p-4 rounded-xl border border-slate-200 hover:border-[#1c2260] hover:bg-blue-50 text-left transition group">
                      <p className="font-semibold text-slate-800 text-sm group-hover:text-[#1c2260]">{opt.label}</p>
                      <p className="text-xs text-slate-400 mt-0.5">{opt.sub}</p>
                    </button>
                  ))}
                </div>
                <button onClick={eligBack} className="h-10 px-4 rounded-xl border border-slate-200 text-sm text-slate-600 hover:bg-slate-50">← Back</button>
              </>
            )}

            {eligStep === 'employment' && (
              <>
                <h2 className="text-xl font-bold text-slate-800 mb-2">Parent/guardian employment</h2>
                <p className="text-sm text-slate-500 mb-6">Determines which income document you'll need to submit.</p>
                <div className="space-y-3 mb-6">
                  {[
                    { v: 'local', label: 'Locally employed', sub: 'Salaried employee in the Philippines' },
                    { v: 'abroad', label: 'OFW / working abroad', sub: 'Overseas Filipino Worker' },
                    { v: 'business', label: 'Self-employed / business owner', sub: 'Entrepreneur, freelancer, or sole proprietor' },
                    { v: 'unemployed', label: 'Unemployed / informal livelihood', sub: 'No formal employer or fixed income' },
                  ].map(opt => (
                    <button key={opt.v} onClick={() => eligGo(nextEligStep('employment', eligAnswers), { employment: opt.v })}
                      className="w-full p-4 rounded-xl border border-slate-200 hover:border-[#1c2260] hover:bg-blue-50 text-left transition group">
                      <p className="font-semibold text-slate-800 text-sm group-hover:text-[#1c2260]">{opt.label}</p>
                      <p className="text-xs text-slate-400 mt-0.5">{opt.sub}</p>
                    </button>
                  ))}
                </div>
                <button onClick={eligBack} className="h-10 px-4 rounded-xl border border-slate-200 text-sm text-slate-600 hover:bg-slate-50">← Back</button>
              </>
            )}

            {eligStep === 'result' && (
              <>
                <h2 className="text-xl font-bold text-slate-800 mb-4">Your ESC Eligibility Result</h2>
                {catResult ? (
                  <>
                    <div className={`p-4 rounded-xl border mb-5 ${catMeta[catResult].tw}`}>
                      <p className="font-bold text-sm">{catMeta[catResult].label}</p>
                      <p className="text-xs mt-1 opacity-80">{catMeta[catResult].desc}</p>
                    </div>
                    <div className="mb-5">
                      <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-3">Documents you'll need</p>
                      <ul className="space-y-2">
                        {docList.map((doc, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                            <FileCheck className="h-4 w-4 text-[#1c2260] shrink-0 mt-0.5" />
                            {doc}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 mb-5">
                      <p className="text-xs text-amber-700">Self-assessment only. The ESC School Committee makes the final eligibility determination.</p>
                    </div>
                    <button onClick={completeEligibility} className="w-full h-12 rounded-xl bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white font-bold text-sm">
                      Continue to Browse Schools →
                    </button>
                  </>
                ) : (
                  <>
                    <div className="p-4 rounded-xl border border-red-200 bg-red-50 mb-5">
                      <p className="font-bold text-sm text-red-700">Not eligible for ESC Subsidy</p>
                      <p className="text-xs text-red-600 mt-1">Households above the middle-class threshold are not eligible for ESC categories B, C, or D. You may still enroll at any private school at full cost.</p>
                    </div>
                    <button
                      onClick={() => { setEligStep('schoolType'); setEligHistory([]); setEligAnswers({ escIntent: true, schoolType: null, segs: [], income: null, employment: null }); }}
                      className="w-full h-10 rounded-xl border border-slate-200 text-sm text-slate-600 hover:bg-slate-50 mb-2"
                    >
                      Start over
                    </button>
                    <button onClick={completeEligibility} className="w-full h-10 rounded-xl bg-slate-100 text-slate-700 font-medium text-sm hover:bg-slate-200">
                      Browse schools without ESC →
                    </button>
                  </>
                )}
                <button onClick={eligBack} className="mt-2 h-9 px-4 text-xs text-slate-400 hover:text-slate-600">← Back</button>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════
  // BROWSE VIEW
  // ═══════════════════════════════════════════════════════════════

  // ── DRAWER TAB CONTENT ────────────────────────────────────────
  const drawerTabContent = {
    status: () => {
      if (appState === 'granted') {
        const cfg = schoolStatusConfigs.granted;
        const name = grantedChoice?.name || 'your chosen school';
        return (
          <div className="p-5 space-y-4">
            <div className={`rounded-xl border p-4 ${cfg.color}`}>
              <div className="flex items-start gap-3">
                {cfg.icon}
                <div>
                  <p className="font-bold text-slate-800 text-sm">{cfg.title}</p>
                  <p className="text-xs text-slate-600 mt-1 leading-relaxed">{cfg.desc(name)}</p>
                  <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                    Enrollment at {name} is a separate, independent process — you may enroll before or after this approval.
                  </p>
                </div>
              </div>
            </div>
          </div>
        );
      }

      if (appState === 'non_esc') {
        const school = wishlist.find(s => s.id === account?.nonEscSchoolId) || schools.find(s => s.id === account?.nonEscSchoolId);
        const name = school?.name || 'your chosen school';
        return (
          <div className="p-5 space-y-4">
            <div className="rounded-xl border p-4 bg-slate-50 border-slate-200">
              <div className="flex items-start gap-3">
                <Info className="h-8 w-8 text-slate-400" />
                <div>
                  <p className="font-bold text-slate-800 text-sm">Enrolled - Non-ESC</p>
                  <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                    You are proceeding with enrollment at {name} without the ESC subsidy.
                  </p>
                </div>
              </div>
            </div>
          </div>
        );
      }

      // appState === 'submitted'
      if (activeChoice) {
        const cfg = schoolStatusConfigs[escStatuses[activeChoice.id]];
        return (
          <div className="p-5 space-y-4">
            <div className={`rounded-xl border p-4 ${cfg.color}`}>
              <div className="flex items-start gap-3">
                {cfg.icon}
                <div>
                  <p className="font-bold text-slate-800 text-sm">{cfg.title}</p>
                  <p className="text-xs text-slate-600 mt-1 leading-relaxed">{cfg.desc(activeChoice.name)}</p>
                </div>
              </div>
            </div>
            {cfg.demo.length > 0 && (
              <div className="rounded-xl border border-dashed border-slate-300 p-4">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-3">Demo Controls</p>
                <div className="space-y-2">
                  {cfg.demo.map(d => (
                    <button key={d.next} onClick={() => advanceSchool(activeChoice.id, d.next)}
                      className={`w-full h-9 rounded-lg text-xs font-bold uppercase tracking-wide ${d.style}`}>
                      {d.label}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      }

      if (lastEngagedChoice && REJECTED_STATES.has(escStatuses[lastEngagedChoice.id])) {
        const cfg = schoolStatusConfigs.rejected;
        return (
          <div className="p-5 space-y-4">
            <div className={`rounded-xl border p-4 ${cfg.color}`}>
              <div className="flex items-start gap-3">
                {cfg.icon}
                <div>
                  <p className="font-bold text-slate-800 text-sm">{cfg.title}</p>
                  <p className="text-xs text-slate-600 mt-1 leading-relaxed">{cfg.desc(lastEngagedChoice.name)}</p>
                </div>
              </div>
            </div>

            {nextChoice && (
              <div className="rounded-xl border border-slate-200 p-4">
                <p className="text-sm text-slate-700 mb-3">Would you like to apply to your next choice, <span className="font-semibold">{nextChoice.name}</span>?</p>
                <button onClick={applyToNextRank} className="w-full h-11 rounded-xl bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white font-semibold text-sm">
                  Yes, Apply to {nextChoice.name} →
                </button>
              </div>
            )}

            <div className="space-y-2">
              <button onClick={() => continueWithoutSubsidy(lastEngagedChoice.id)} className="w-full h-11 rounded-xl bg-slate-800 text-white font-semibold text-sm">
                Continue Enrollment at {lastEngagedChoice.name} (No Subsidy) →
              </button>
              {!nextChoice && (
                <button onClick={applyAgainDifferentSchool} className="w-full h-11 rounded-xl border border-slate-200 text-slate-700 font-medium text-sm hover:bg-slate-50">
                  Stop and Choose Different Private Schools →
                </button>
              )}
            </div>
          </div>
        );
      }

      return null;
    },

    choices: () => (
      <div className="p-5 space-y-3">
        {isPostSubmission && (
          <div className="rounded-lg bg-slate-50 border border-slate-200 p-3">
            <p className="text-xs text-slate-500">Your school choices have been submitted and are read-only.</p>
          </div>
        )}
        {wishlist.length === 0 && (
          <div className="text-center py-12">
            <Heart className="h-8 w-8 text-slate-200 mx-auto mb-3" />
            <p className="text-sm text-slate-400">No schools added yet.</p>
            <p className="text-xs text-slate-300 mt-1">Browse the map and click + to add schools.</p>
          </div>
        )}
        {wishlist.map((school, i) => {
          const escStatus = escStatuses[school.id];
          return (
            <div key={school.id} className="flex items-start gap-3 p-3 rounded-xl border border-slate-200 bg-white">
              <span className="h-6 w-6 shrink-0 rounded-full bg-[#1c2260]/10 text-[#1c2260] text-xs font-bold flex items-center justify-center">{i + 1}</span>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-slate-800 leading-snug">{school.name}</p>
                <p className="text-xs text-slate-400 mt-0.5">{school.municipality} · {school.type === 'public' ? 'Public' : school.esc_subsidy > 0 ? 'Private ESC' : 'Private'}</p>
                {escStatus && (
                  <span className={`inline-block mt-1.5 text-[9px] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded border ${schoolStatusConfigs[escStatus]?.color || 'bg-slate-50 text-slate-400 border-slate-200'}`}>
                    {schoolStatusConfigs[escStatus]?.title || escStatus}
                  </span>
                )}
              </div>
              {!isPostSubmission && (
                <button onClick={() => removeFromWishlist(school.id)} className="text-slate-300 hover:text-red-400 shrink-0">
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          );
        })}
        {!isPostSubmission && !hasPublicAlternative && wishlist.length > 0 && (
          <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 text-xs text-amber-700">
            Add at least one public JHS to ensure placement.
          </div>
        )}
      </div>
    ),

    documents: () => {
      const isActiveDocsPending = activeChoice && escStatuses[activeChoice.id] === 'docs_pending';

      // Post-submission but no additional docs requested — show review status only
      if (isPostSubmission && !isActiveDocsPending) {
        return (
          <div className="p-5 space-y-4">
            <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
              <div className="flex items-start gap-3">
                <FileCheck className="h-5 w-5 text-blue-500 shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-blue-800 text-sm">Documents Submitted</p>
                  <p className="text-xs text-blue-600 mt-1 leading-relaxed">
                    The school committee is currently reviewing your application documents. You will be notified if additional documents are required.
                  </p>
                </div>
              </div>
            </div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Submitted Documents</p>
            <div className="space-y-2">
              {(account?.uploadedDocs || []).map((doc, i) => (
                <div key={i} className="flex items-start gap-3 p-3 rounded-xl border border-green-200 bg-green-50">
                  <div className="h-5 w-5 rounded-full shrink-0 flex items-center justify-center mt-0.5 bg-green-500">
                    <Check className="h-3 w-3 text-white" strokeWidth={3} />
                  </div>
                  <p className="text-xs text-slate-700 leading-snug">{doc}</p>
                </div>
              ))}
            </div>
          </div>
        );
      }

      // docs_pending or pre-submission — show upload interface
      return (
        <div className="p-5">
          {appState === 'not_eligible' ? (
            <p className="text-sm text-slate-500">You're not eligible for the ESC subsidy, so no ESC documents are required. You can enroll directly through the standard DepEd pathway.</p>
          ) : !account?.category ? (
            <p className="text-sm text-slate-500">Complete your eligibility assessment first to see your required documents.</p>
          ) : (
            <>
              {isActiveDocsPending && (
                <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 mb-4 text-xs text-amber-800 leading-relaxed">
                  {activeChoice.name}'s school committee has requested an additional document. Please upload it below.
                </div>
              )}
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-4">Required Documents — Category {account.category}</p>
              <div className="space-y-3">
                {requiredDocs.map((doc, i) => {
                  const uploaded = uploadedDocs.includes(doc);
                  return (
                    <div key={i} className={`flex items-start gap-3 p-3 rounded-xl border ${uploaded ? 'border-green-200 bg-green-50' : 'border-slate-200 bg-white'}`}>
                      <div className={`h-5 w-5 rounded-full shrink-0 flex items-center justify-center mt-0.5 ${uploaded ? 'bg-green-500' : 'border-2 border-slate-300'}`}>
                        {uploaded && <Check className="h-3 w-3 text-white" strokeWidth={3} />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs text-slate-700 leading-snug">{doc}</p>
                        {!uploaded && (
                          <button
                            onClick={() => {
                              updateAccount({ uploadedDocs: [...uploadedDocs, doc] });
                            }}
                            className="mt-1.5 text-[10px] font-bold text-[#1c2260] uppercase tracking-wide hover:underline"
                          >
                            Simulate Upload ↑
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
              {!docsReady && (
                <button
                  onClick={() => { updateAccount({ uploadedDocs: requiredDocs }); }}
                  className="mt-4 w-full h-9 rounded-xl border border-slate-200 text-xs text-slate-500 hover:bg-slate-50"
                >
                  Simulate all uploads (demo)
                </button>
              )}
              {docsReady && (
                <div className="mt-4 rounded-lg bg-green-50 border border-green-200 p-3 text-xs text-green-700 text-center font-medium">
                  All documents ready ✓
                </div>
              )}
              {isActiveDocsPending && docsReady && (
                <button onClick={() => { advanceSchool(activeChoice.id, 'docs_submitted'); setDrawerTab('status'); }} className="mt-3 w-full h-11 rounded-xl bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white font-semibold text-sm">
                  Submit Additional Document →
                </button>
              )}
            </>
          )}
        </div>
      );
    },

    survey: () => !isPostSubmission ? (
      <div className="p-5 space-y-6">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">Pilot Survey</p>
          <p className="text-xs text-slate-400">
            {appState === 'not_eligible' ? '2 required questions before you can enroll.' : '3 required questions before you can submit.'}
          </p>
        </div>

        <div className="space-y-5">
          <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Using PAARAL</p>

          <div>
            <p className="text-sm font-semibold text-slate-800 mb-3">1. How easy was it to find schools?</p>
            <div className="flex gap-2">
              {[1,2,3,4,5].map(n => (
                <button key={n} onClick={() => setSurveyAnswers(a => ({ ...a, ease: n }))}
                  className={`flex-1 h-9 rounded-lg border text-sm font-bold transition ${surveyAnswers.ease === n ? 'bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] border-[#1c2260] text-white' : 'border-slate-200 text-slate-500 hover:border-slate-300'}`}>
                  {n}
                </button>
              ))}
            </div>
            <div className="flex justify-between text-[10px] text-slate-400 mt-1 px-0.5">
              <span>Very hard</span><span>Very easy</span>
            </div>
          </div>

          <div>
            <p className="text-sm font-semibold text-slate-800 mb-3">2. Did this information help you decide where to enroll?</p>
            <div className="space-y-2">
              {['Yes', 'Somewhat', 'No'].map(opt => (
                <button key={opt} onClick={() => setSurveyAnswers(a => ({ ...a, helpful: opt }))}
                  className={`w-full h-10 rounded-xl border text-sm text-left px-4 transition ${surveyAnswers.helpful === opt ? 'bg-blue-50 border-[#1c2260] text-[#1c2260] font-semibold' : 'border-slate-200 text-slate-600 hover:border-slate-300'}`}>
                  {opt}
                </button>
              ))}
            </div>
          </div>
        </div>

        {appState !== 'not_eligible' && (
          <div className="space-y-3">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">About Your ESC Application</p>
            <div>
              <p className="text-sm font-semibold text-slate-800 mb-3">3. Biggest concern about enrolling in a private school through ESC?</p>
              <div className="space-y-2">
                {['Cost', 'Distance', 'School quality', 'Slot availability'].map(opt => (
                  <button key={opt} onClick={() => setSurveyAnswers(a => ({ ...a, concern: opt }))}
                    className={`w-full h-10 rounded-xl border text-sm text-left px-4 transition ${surveyAnswers.concern === opt ? 'bg-blue-50 border-[#1c2260] text-[#1c2260] font-semibold' : 'border-slate-200 text-slate-600 hover:border-slate-300'}`}>
                    {opt}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="rounded-xl border border-slate-200 p-4 space-y-2">
          <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">
            {appState === 'not_eligible' ? 'Enrollment Checklist' : 'Submission Checklist'}
          </p>
          {(appState === 'not_eligible'
            ? [
                { done: wishlist.length > 0, label: 'At least one school added' },
                { done: generalSurveyComplete, label: 'Survey complete' },
              ]
            : [
                { done: hasPrivateChoice, label: 'At least one private school added' },
                { done: hasPublicAlternative, label: 'Public JHS included (guaranteed fallback)' },
                { done: docsReady, label: 'Documents uploaded' },
                { done: generalSurveyComplete && escSurveyComplete, label: 'Survey complete' },
              ]
          ).map(({ done, label }) => (
            <div key={label} className={`flex items-center gap-2 text-xs ${done ? 'text-green-700' : 'text-slate-400'}`}>
              <span className={`h-4 w-4 rounded-full flex items-center justify-center shrink-0 ${done ? 'bg-green-500' : 'bg-slate-200'}`}>
                {done && <Check className="h-2.5 w-2.5 text-white" strokeWidth={3} />}
              </span>
              {label}
            </div>
          ))}
        </div>

        {appState === 'not_eligible' ? (
          <button onClick={handleEnrollNonEsc} disabled={!canEnrollNonEsc}
            className="w-full h-12 rounded-xl bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white font-bold text-sm disabled:opacity-40 disabled:cursor-not-allowed">
            Enroll Without ESC
          </button>
        ) : (
          <button onClick={handleSubmitEsc} disabled={!canSubmitEsc}
            className="w-full h-12 rounded-xl bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white font-bold text-sm disabled:opacity-40 disabled:cursor-not-allowed">
            Submit Application
          </button>
        )}
      </div>
    ) : null,
  };

  // School profile view content (JSX variable, depends on selectedSchool)
  const profileViewContent = selectedSchool ? (() => {
    const s = selectedSchool;
    const escP = s.esc_subsidy > 0;
    return (
      <div className="flex flex-col h-full">
        <div className="sticky top-0 z-10 bg-white border-b border-slate-200 flex">
          {[
            { id: 'overview', label: 'Overview', ref: overviewRef },
            { id: 'characteristics', label: 'Details', ref: characteristicsRef },
            { id: 'fee', label: 'Fees', ref: feeRef },
          ].map(({ id, label, ref }) => (
            <button key={id} onClick={() => scrollToSection(ref)}
              className={`flex-1 py-3 text-xs font-bold uppercase tracking-widest transition ${profileSection === id ? 'border-b-2 border-[#1c2260] text-[#1c2260]' : 'text-slate-400 hover:text-slate-600'}`}>
              {label}
            </button>
          ))}
        </div>
        <div ref={profileScrollRef} className="flex-1 overflow-y-auto custom-scrollbar">
          <div ref={overviewRef} className="p-6 border-b border-slate-100">
            <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${typeMeta[s.type].badge}`}>{typeMeta[s.type].label}</span>
            <h2 className="text-lg font-bold text-slate-800 mt-2 leading-tight">{s.name}</h2>
            <p className="text-sm text-slate-500 mt-1">{s.barangay}, {s.municipality}, {s.province}</p>
            <div className="mt-4 rounded-xl bg-slate-50 h-40 flex items-center justify-center border border-slate-200">
              <p className="text-xs text-slate-400">Gallery placeholder</p>
            </div>
            <p className="mt-4 text-sm text-slate-600 leading-relaxed">
              {s.name} is a {typeMeta[s.type].label.toLowerCase()} junior high school located in {s.municipality}.
              {escP ? ` It participates in the DepEd ESC program, offering a subsidy of ${pesos(s.esc_subsidy)}/year for eligible learners.` : ''}
            </p>
          </div>
          <div ref={characteristicsRef} className="p-6 border-b border-slate-100">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-4">School Characteristics</p>
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'Region', value: s.region },
                { label: 'Admission', value: s.admission_category },
                { label: 'Sector', value: s.religious_affiliation },
                { label: 'ESC Rating', value: s.esc_rating ? `${s.esc_rating}/5` : 'N/A' },
                { label: 'Distance', value: `${s.distance_km} km` },
                { label: 'Commute', value: `~${s.commute_minutes} min` },
              ].map(({ label, value }) => (
                <div key={label}>
                  <p className="text-[11px] uppercase tracking-wide text-slate-400">{label}</p>
                  <p className="text-sm font-semibold text-slate-800 mt-0.5">{value}</p>
                </div>
              ))}
            </div>
          </div>
          <div ref={feeRef} className="p-6">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-4">Fee Information</p>
            <div className="grid grid-cols-3 gap-3 mb-4">
              {[
                { label: 'Tuition', value: pesos(s.tuition), cls: 'text-slate-700' },
                { label: 'ESC Subsidy', value: escP ? `-${pesos(s.esc_subsidy)}` : 'None', cls: escP ? 'text-green-600' : 'text-slate-400' },
                { label: 'Net Cost', value: pesos(s.net_cost), cls: 'text-[#1c2260]' },
              ].map(({ label, value, cls }) => (
                <div key={label} className="rounded-xl border border-slate-200 p-3 text-center">
                  <p className="text-[10px] uppercase tracking-wide text-slate-400">{label}</p>
                  <p className={`text-sm font-bold mt-1 ${cls}`}>{value}</p>
                </div>
              ))}
            </div>
            {escP && (
              <div className="rounded-xl bg-green-50 border border-green-200 p-3 text-xs text-green-700 mb-4">
                ESC partner — eligible learners get <strong>{pesos(s.esc_subsidy)}</strong>/year subsidy, reducing cost to <strong>{pesos(s.net_cost)}</strong>.
              </div>
            )}
            {(escP || s.type === 'public') && (
              <div className="mb-4">
                <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
                  <span>Available slots: {s.slots_available} of {s.slots_total}</span>
                  <span>{pct(s.slots_available, s.slots_total)}%</span>
                </div>
                <div className="h-2 rounded-full bg-slate-200 overflow-hidden">
                  <div className={`h-full rounded-full ${slotTone(s)}`} style={{ width: `${pct(s.slots_available, s.slots_total)}%` }} />
                </div>
              </div>
            )}
            <WishlistButton school={s} isInList={isInWishlist(s)} onAdd={addToWishlist} />
          </div>
        </div>
      </div>
    );
  })() : (
    <div className="flex flex-col items-center justify-center h-full text-center p-8">
      <MapPin className="h-10 w-10 text-slate-200 mb-4" />
      <p className="text-slate-400 text-sm">Select a school on the map or from the list to view its profile.</p>
    </div>
  );

  // ── MAIN BROWSE RENDER ────────────────────────────────────────
  return (
    <div className="fixed inset-0">
      <link href="https://fonts.googleapis.com/css2?family=Baskervville&display=swap" rel="stylesheet" />
      {loginModal}

      {/* Map fills the full viewport */}
      <PhilippinesMap
        filteredSchools={filteredSchools}
        selectedSchool={selectedSchool}
        hoveredId={hoveredId}
        onHover={setHoveredId}
        onSelect={handleMapSelect}
        userLocation={userLocation}
      />

      {/* Top-left floating bar */}
      <div className="absolute top-4 left-4 z-30 flex items-center gap-2">
        <button
          type="button"
          onClick={() => setLeftPanel(p => p === 'filters' ? null : 'filters')}
          className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-md hover:bg-slate-50 transition"
        >
          {leftPanel === 'filters'
            ? <X className="h-4 w-4 text-slate-600" />
            : <Menu className="h-4 w-4 text-slate-600" />}
        </button>
        <div className="flex h-10 w-72 items-center gap-2 rounded-full bg-white pl-4 pr-3 shadow-md">
          <Search className="h-4 w-4 flex-shrink-0 text-slate-400" />
          <input
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            placeholder="Search schools…"
            className="min-w-0 flex-1 bg-transparent text-sm text-slate-800 outline-none placeholder:text-slate-400"
          />
          {searchTerm && (
            <button type="button" onClick={() => setSearchTerm('')}>
              <X className="h-3.5 w-3.5 text-slate-400 hover:text-slate-600" />
            </button>
          )}
        </div>
        {account ? (
          <button
            type="button"
            onClick={() => setDrawerOpen(o => !o)}
            className="flex h-10 items-center gap-2 rounded-full bg-white px-4 text-sm font-semibold text-slate-700 shadow-md hover:bg-slate-50 transition"
          >
            <User className="h-4 w-4 flex-shrink-0" />
            My Account
            {wishlist.length > 0 && !isPostSubmission && (
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-[9px] font-bold text-white">
                {wishlist.length}
              </span>
            )}
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setShowLogin(true)}
            className="h-10 rounded-full bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] px-5 text-sm font-semibold text-white shadow-md hover:opacity-90 transition"
          >
            Log In
          </button>
        )}
      </div>

      {/* PAARAL branding — bottom-left, visible only when panel is closed */}
      {!leftPanel && (
        <div className="absolute bottom-5 left-5 z-20 select-none">
          <p className="font-['Baskervville',serif] text-2xl tracking-widest text-white drop-shadow-lg">
            PROJECT PAARAL
          </p>
          <p className="mt-0.5 text-[11px] font-medium text-white/60 tracking-wide drop-shadow">
            Platform for Analyzing Access and Resource Allocation in Learning
          </p>
        </div>
      )}

      {/* School type filter chips */}
      <div className="absolute z-20 flex gap-2" style={{ top: '72px', left: '60px' }}>
        {Object.entries(typeMeta).map(([type, meta]) => (
          <button
            key={type}
            type="button"
            onClick={() => setSchoolTypes(ts => ts.includes(type) ? ts.filter(t => t !== type) : [...ts, type])}
            className={[
              'flex h-8 items-center gap-1.5 rounded-full px-3 text-[11px] font-semibold shadow-md transition',
              schoolTypes.includes(type)
                ? 'bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] text-white'
                : 'bg-white text-slate-600 hover:bg-slate-50',
            ].join(' ')}
          >
            <span className="h-2 w-2 flex-shrink-0 rounded-full" style={{ backgroundColor: meta.dot }} />
            {meta.label}
            <span className={`tabular-nums text-[10px] ${schoolTypes.includes(type) ? 'text-white/70' : 'text-slate-400'}`}>
              {filteredSchools.filter(s => s.type === type).length}
            </span>
          </button>
        ))}
      </div>

      {/* Left panel — filters or school detail */}
      {leftPanel && (
        <div className="absolute inset-y-0 left-0 z-40 flex w-[360px] flex-col bg-white shadow-2xl">
          {leftPanel === 'filters' && (
            <>
              {/* Tab bar */}
              <div className="flex shrink-0 items-center border-b border-slate-100">
                {[{ id: 'about', label: 'About PAARAL' }, { id: 'find', label: 'Find a School' }].map(t => (
                  <button key={t.id} type="button" onClick={() => setLeftPanelTab(t.id)}
                    className={[
                      'flex-1 py-3.5 text-[11px] font-bold uppercase tracking-wider transition',
                      leftPanelTab === t.id
                        ? 'border-b-2 border-[#1c2260] text-[#1c2260]'
                        : 'text-slate-400 hover:text-slate-600',
                    ].join(' ')}>
                    {t.label}
                  </button>
                ))}
                <button type="button" onClick={() => setLeftPanel(null)} className="mr-3 flex h-7 w-7 shrink-0 items-center justify-center rounded-full hover:bg-slate-100">
                  <X className="h-4 w-4 text-slate-400" />
                </button>
              </div>

              {/* Tab content */}
              <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar">

                {leftPanelTab === 'about' && (
                  <div className="p-6 space-y-5">
                    <div>
                      <p className="font-['Baskervville',serif] text-xl tracking-widest text-[#1a1d23]">PROJECT PAARAL</p>
                      <p className="mt-1 text-[11px] font-medium uppercase tracking-widest text-slate-400">
                        Platform for Analyzing Access and Resource Allocation in Learning
                      </p>
                    </div>
                    <p className="text-sm leading-6 text-slate-600">
                      Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
                    </p>
                    <p className="text-sm leading-6 text-slate-600">
                      Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                    </p>
                    <p className="text-sm leading-6 text-slate-600">
                      Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.
                    </p>
                  </div>
                )}

                {leftPanelTab === 'find' && (
                  <>
                    <div className="space-y-1 p-4">
                      <FilterSection title="Location" id="location" open={openSections.location} onToggle={toggleSection}>
                        <div className="flex flex-col gap-2">
                          <SelectField value={region} onChange={setRegion} options={regionOptions} placeholder="All Regions" />
                          <SelectField value={province} onChange={setProvince} options={provinceOptions} placeholder="All Provinces" />
                          <SelectField value={municipality} onChange={setMunicipality} options={municipalityOptions} placeholder="All Municipalities" />
                          <SelectField value={barangay} onChange={setBarangay} options={barangayOptions} placeholder="All Barangays" />
                        </div>
                      </FilterSection>
                      <FilterSection title="Distance (km)" id="distance" open={openSections.distance} onToggle={toggleSection}>
                        <RangePair min={0} max={100} value={distance} onChange={setDistance} format={v => `${v} km`} />
                      </FilterSection>
                      <FilterSection title="Net Cost" id="tuition" open={openSections.tuition} onToggle={toggleSection}>
                        <RangePair min={0} max={250000} step={5000} value={tuition} onChange={setTuition} format={v => v === 0 ? 'Free' : `₱${(v/1000).toFixed(0)}k`} />
                      </FilterSection>
                      <FilterSection title="School Type" id="type" open={openSections.type} onToggle={toggleSection}>
                        {Object.entries(typeMeta).map(([type, meta]) => (
                          <CheckboxRow key={type} checked={schoolTypes.includes(type)} label={meta.label}
                            sublabel={`${filteredSchools.filter(s => s.type === type).length}`}
                            onChange={() => setSchoolTypes(ts => ts.includes(type) ? ts.filter(t => t !== type) : [...ts, type])} />
                        ))}
                      </FilterSection>
                      <FilterSection title="Commute Time" id="commute" open={openSections.commute} onToggle={toggleSection}>
                        {[{ v: 'under5', l: 'Under 5 min' }, { v: '15to30', l: '15–30 min' }, { v: 'over30', l: 'Over 30 min' }].map(b => (
                          <CheckboxRow key={b.v} checked={commuteBuckets.includes(b.v)} label={b.l}
                            onChange={() => setCommuteBuckets(bs => bs.includes(b.v) ? bs.filter(x => x !== b.v) : [...bs, b.v])} />
                        ))}
                      </FilterSection>
                    </div>
                    <div className="border-t border-slate-100 px-3 pb-4 pt-1">
                      <p className="px-1 py-2 text-[10px] font-bold uppercase tracking-widest text-slate-400">
                        {filteredSchools.length} results
                      </p>
                      <div ref={resultsScrollRef} className="space-y-2">
                        {filteredSchools.map(school => (
                          <div key={school.id} ref={selectedSchool?.id === school.id ? selectedCardRef : null}>
                            <ResultCard
                              school={school}
                              selected={selectedSchool?.id === school.id}
                              onSelect={(s) => { setSelectedSchool(s); setLeftPanel('school'); }}
                              onAddToWishlist={addToWishlist}
                              isInWishlist={isInWishlist(school)}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {leftPanel === 'school' && selectedSchool && (
            <>
              <div className="flex h-14 shrink-0 items-center gap-2 border-b border-slate-100 px-4">
                <button type="button" onClick={() => setLeftPanel('filters')} className="flex h-8 w-8 items-center justify-center rounded-full hover:bg-slate-100">
                  <ChevronLeft className="h-4 w-4 text-slate-500" />
                </button>
                <p className="flex-1 truncate text-sm font-bold text-slate-800">{selectedSchool.name}</p>
                <button type="button" onClick={() => { setLeftPanel(null); setSelectedSchool(null); }} className="flex h-8 w-8 items-center justify-center rounded-full hover:bg-slate-100">
                  <X className="h-4 w-4 text-slate-400" />
                </button>
              </div>
              <div ref={profileScrollRef} className="flex-1 overflow-y-auto custom-scrollbar">
                {profileViewContent}
              </div>
            </>
          )}
        </div>
      )}

      {/* Application Drawer */}
      {drawerOpen && (
        <div className="absolute inset-0 z-30 flex">
          <div className="flex-1 bg-black/30" onClick={() => setDrawerOpen(false)} />
          <div className="flex h-full w-[360px] flex-col bg-white shadow-2xl">
            <div className="flex h-14 shrink-0 items-center justify-between border-b border-slate-100 px-5">
              <p className="text-sm font-bold text-slate-800">My Application</p>
              <button type="button" onClick={() => setDrawerOpen(false)} className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-200 text-slate-400 hover:bg-slate-50">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex flex-1 min-h-0 flex-col overflow-hidden">
              {!account ? (
                <div className="flex flex-1 flex-col items-center justify-center p-8 text-center">
                  <User className="mb-4 h-10 w-10 text-slate-300" />
                  <p className="mb-4 text-sm text-slate-500">Log in to track your application.</p>
                  <button type="button" onClick={() => { setDrawerOpen(false); setShowLogin(true); }} className="h-10 rounded-xl bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] px-5 text-sm font-semibold text-white">
                    Log In
                  </button>
                </div>
              ) : (
                <>
                  <div className="shrink-0 border-b border-slate-100 px-5 pb-3 pt-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-bold text-slate-800">{account.name}</p>
                        <p className="font-mono text-xs text-slate-400">LRN: {account.lrn}</p>
                      </div>
                      {account.category && (
                        <span className={`rounded border px-2 py-0.5 text-[10px] font-bold uppercase ${catMeta[account.category]?.tw}`}>
                          Cat. {account.category}
                        </span>
                      )}
                    </div>
                    {POST_SUBMISSION_STATES.has(account.applicationState) && (
                      <div className="mt-2">{renderStateBadge(account.applicationState)}</div>
                    )}
                  </div>
                  <div className="flex shrink-0 border-b border-slate-100">
                    {drawerTabList.map(tab => (
                      <button key={tab} type="button" onClick={() => setDrawerTab(tab)}
                        className={`flex-1 py-3 text-[10px] font-bold uppercase tracking-wider capitalize transition ${drawerTab === tab ? 'border-b-2 border-[#1c2260] text-[#1c2260]' : 'text-slate-400 hover:text-slate-600'}`}>
                        {tab}
                        {tab === 'choices' && wishlist.length > 0 && (
                          <span className="ml-1 rounded-full bg-gradient-to-br from-[#1c2260] via-[#5a2d68] to-[#c44820] px-1.5 py-0.5 text-[9px] text-white">{wishlist.length}</span>
                        )}
                      </button>
                    ))}
                  </div>
                  <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {drawerTabContent[drawerTab]?.()}
                  </div>
                </>
              )}
            </div>
            {account && (
              <div className="shrink-0 border-t border-slate-100 px-5 py-3">
                <button type="button" onClick={logout} className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-600">
                  <LogOut className="h-3.5 w-3.5" /> Log out
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
