"use client";

import Image from "next/image";
import Link from "next/link";
import Button from "@mui/material/Button";
import { User } from "lucide-react";
import { useApplication } from "@/components/templates/ApplicationStateProvider";

/** Top header — logo lockup, the primary "Browse Schools" CTA, and an
 * account control (Log In, or "My Account" once logged in — a link to the
 * single standalone /account page, which holds identity plus the full
 * Status/Choices/Documents/Survey workspace). Present on every page so it's
 * always the same predictable way back to the home page (via the logo) and
 * into the application. Nav links are deliberately not fabricated yet — add
 * them once real destination pages exist. */
export default function SiteHeader() {
  const { account, openLoginModal } = useApplication();

  return (
    <header className="flex items-center justify-between bg-[image:var(--linearPrimaryAccent)] px-6 py-4 md:px-12">
      <Link href="/" className="flex items-center gap-4">
        <Image
          src="/assets/deped-logo.png"
          alt="DepEd"
          width={32}
          height={32}
          className="h-8 w-auto"
        />
        <Image
          src="/assets/ecair-logo.png"
          alt="ECAIR"
          width={20}
          height={20}
          className="h-5 w-auto"
        />
        <span className="text-xl font-bold tracking-tight text-white">
          PAARAL
        </span>
      </Link>
      <div className="flex items-center gap-3">
        <Button
          component={Link}
          href="/browse"
          variant="contained"
          sx={{
            bgcolor: "var(--background)",
            color: "var(--primary)",
            fontWeight: 700,
            "&:hover": { bgcolor: "var(--background)", opacity: 0.9 },
          }}
        >
          Browse Schools
        </Button>
        {account ? (
          <Button
            component={Link}
            href="/account"
            variant="outlined"
            startIcon={<User className="h-4 w-4" />}
            sx={{
              borderColor: "rgba(255,255,255,0.4)",
              color: "white",
              "&:hover": { borderColor: "white" },
            }}
          >
            My Account
          </Button>
        ) : (
          <Button
            onClick={openLoginModal}
            variant="outlined"
            sx={{
              borderColor: "rgba(255,255,255,0.4)",
              color: "white",
              "&:hover": { borderColor: "white" },
            }}
          >
            Log In
          </Button>
        )}
      </div>
    </header>
  );
}
