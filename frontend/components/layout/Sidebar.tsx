"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { checkHealth } from "@/lib/api";

const NAV_ITEMS = [
  {
    path: "/universal-restoration",
    label: "Universal Restoration",
    icon: "auto_fix_high",
  },
  {
    path: "/hard-routing",
    label: "Hard-Routed Restoration",
    icon: "alt_route",
  },
  {
    path: "/soft-mixture",
    label: "Soft Mixture-of-Experts",
    icon: "layers",
  },
  {
    path: "/face-to-sketch",
    label: "Face-to-Sketch Generator",
    icon: "draw",
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const [backendStatus, setBackendStatus] = useState<"ok" | "offline" | "checking">("checking");

  useEffect(() => {
    let mounted = true;
    checkHealth()
      .then((res) => {
        if (mounted) {
          setBackendStatus(res.status === "ok" ? "ok" : "offline");
        }
      })
      .catch(() => {
        if (mounted) setBackendStatus("offline");
      });

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <aside className="fixed left-0 top-0 h-full w-[240px] bg-[#F5F5F7] border-r border-[#D2D2D7] z-50 flex flex-col justify-between select-none">
      <div className="flex flex-col">
        {/* Brand Header with Precision Vision Studio Logo */}
        <div className="px-4 py-4 border-b border-[#D2D2D7] flex items-center gap-2.5">
          <div className="w-8 h-8 shrink-0 bg-[#0071E3] flex items-center justify-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 40 40"
              className="w-7 h-7"
              fill="none"
            >
              <rect x="8" y="8" width="10" height="10" fill="#FFFFFF" />
              <rect x="22" y="8" width="10" height="10" fill="#FFFFFF" fillOpacity="0.4" />
              <rect x="8" y="22" width="10" height="10" fill="#FFFFFF" fillOpacity="0.4" />
              <rect x="22" y="22" width="10" height="10" fill="#FFFFFF" />
              <path d="M13 13L27 27" stroke="#0071E3" strokeWidth="2" />
            </svg>
          </div>
          <div className="flex flex-col min-w-0">
            <span className="text-[14px] font-semibold text-[#1D1D1F] tracking-tight truncate leading-tight">
              Precision Vision
            </span>
            <span className="text-[9px] font-semibold uppercase tracking-wider text-[#6E6E73] truncate">
              Vision Benchmark Lab
            </span>
          </div>
        </div>

        {/* Navigation list */}
        <nav className="flex flex-col mt-2">
          {NAV_ITEMS.map((item) => {
            const isActive =
              pathname === item.path ||
              (item.path === "/universal-restoration" && pathname === "/");

            return (
              <Link
                key={item.path}
                href={item.path}
                className={`flex items-center gap-3 py-2.5 px-4 transition-colors text-[13px] border-l-[3px] ${
                  isActive
                    ? "border-l-[#0071E3] bg-[#E8E8EA] text-[#1D1D1F] font-semibold"
                    : "border-l-transparent text-[#6E6E73] hover:bg-[#EEEEEF] hover:text-[#1D1D1F] font-normal"
                }`}
              >
                <span className="material-symbols-outlined text-[18px] shrink-0">
                  {item.icon}
                </span>
                <span className="truncate">{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer System Telemetry Status */}
      <div className="p-3 border-t border-[#D2D2D7] bg-[#F5F5F7]">
        <div className="flex items-center justify-between text-[11px] font-mono-code text-[#6E6E73]">
          <div className="flex items-center gap-1.5 truncate">
            <span
              className={`w-2 h-2 shrink-0 inline-block ${
                backendStatus === "ok"
                  ? "bg-[#1F8A3B]"
                  : backendStatus === "checking"
                  ? "bg-[#B8860B] animate-pulse"
                  : "bg-[#0071E3]"
              }`}
            />
            <span className="truncate">
              {backendStatus === "ok"
                ? "Backend: Live"
                : backendStatus === "checking"
                ? "Checking Hub"
                : "Active Mode: Demo/API"}
            </span>
          </div>
          <span className="text-[10px] text-[#A1A1A6]">v2.0</span>
        </div>
      </div>
    </aside>
  );
}
