"use client";

import { useEffect } from "react";

/**
 * Scrolls instantly to the section matching the URL's hash on mount.
 * Works around a real, reproducible bug: a plain `<Link href="/#about-esc">`
 * clicked from another page (e.g. SiteHeader's nav on `/browse`) loads `/`
 * but never actually scrolls to the target section — the browser/Next
 * router's built-in hash-anchor scroll silently no-ops here, the same class
 * of unreliable-smooth-scroll issue already documented and worked around in
 * `HeroSection`'s own `scrollToSection` (`behavior: "instant"` is the only
 * behavior confirmed reliable in this app). Renders nothing.
 */
export default function ScrollToHash() {
  useEffect(() => {
    const hash = window.location.hash;
    if (!hash) return;
    document
      .getElementById(hash.slice(1))
      ?.scrollIntoView({ behavior: "instant", block: "start" });
  }, []);

  return null;
}
