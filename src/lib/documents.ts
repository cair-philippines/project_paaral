import { apiDelete, apiGet, apiUpload } from "@/lib/api";

/** One stored document, as `paaral-student-api`'s document endpoints
 * (Chunk 18) return it - camelCase, matching the `CamelModel` convention
 * already used for wishlist/survey/status. */
export interface ApiDocument {
  documentType: string;
  fileUrl: string;
  uploadedAt: string;
}

/** Accepted file types for a document upload - mirrors
 * `ALLOWED_CONTENT_TYPES` in `paaral-student-api`'s
 * `app/services/document.py` exactly, so the frontend rejects an
 * unsupported file before ever making a network call. Keep the two in
 * sync if either changes. */
export const ALLOWED_DOCUMENT_TYPES = [
  "image/jpeg",
  "image/png",
  "application/pdf",
];

/** Per-file size cap - mirrors `settings.max_upload_size_mb` (10MB) on
 * the backend. Client-side enforcement is a courtesy (instant feedback
 * before wasting an upload); the backend re-checks regardless, since a
 * client-side-only check can always be bypassed. */
export const MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024;

/** Return every document a learner has on file. */
export function listDocuments(lrn: string): Promise<ApiDocument[]> {
  return apiGet<ApiDocument[]>(`/api/v1/applications/${lrn}/documents`);
}

/** Upload a file into one document-type "bin," replacing any file
 * already there. `documentType` is free text (e.g. "Certificate of
 * Indigency," from `getDocList()`) - URL-encoded here since it may
 * contain spaces. */
export function uploadDocument(
  lrn: string,
  documentType: string,
  file: File
): Promise<ApiDocument> {
  return apiUpload<ApiDocument>(
    `/api/v1/applications/${lrn}/documents/${encodeURIComponent(documentType)}`,
    file
  );
}

/** Clear a document-type bin. */
export function deleteDocument(
  lrn: string,
  documentType: string
): Promise<void> {
  return apiDelete(
    `/api/v1/applications/${lrn}/documents/${encodeURIComponent(documentType)}`
  );
}
