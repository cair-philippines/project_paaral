"use client";

import type { ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";

interface ScrollRevealProps {
  children: ReactNode;
  delay?: number;
  className?: string;
}

/**
 * Gentle "appear on scroll" entrance — a short upward fade, triggered once
 * when a section enters the viewport. This is a standard reveal, not
 * scroll-jacking: it never locks or hijacks the user's scroll position,
 * it only animates opacity/position of content that would be visible at
 * that scroll offset anyway. Timing (220ms, ease-out) matches the
 * SchoolPath reference's stated animation philosophy ("180-240ms upward
 * fade, only when entering a view").
 *
 * Respects `prefers-reduced-motion`: renders children in their final state
 * immediately, with no animation at all, per this project's accessibility
 * floor (WCAG 2.1 AA + the ui-ux-designer agent's low-tech-literacy /
 * motion-sensitivity guidance).
 */
export default function ScrollReveal({
  children,
  delay = 0,
  className,
}: ScrollRevealProps) {
  const reduceMotion = useReducedMotion();

  if (reduceMotion) {
    return <div className={className}>{children}</div>;
  }

  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.22, delay, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
}
