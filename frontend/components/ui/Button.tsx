"use client";

import React, { ButtonHTMLAttributes, forwardRef } from "react";

export type ButtonVariant = "primary" | "secondary" | "outline" | "destructive" | "ghost";
export type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  loadingText?: string;
  icon?: React.ReactNode;
  iconRight?: React.ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      variant = "primary",
      size = "md",
      loading = false,
      loadingText,
      disabled,
      icon,
      iconRight,
      className = "",
      ...props
    },
    ref
  ) => {
    const baseStyles =
      "inline-flex items-center justify-center font-semibold uppercase tracking-wider transition-colors select-none rounded-none focus:outline-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-60";

    const sizeStyles = {
      sm: "h-7 px-2.5 text-[11px] gap-1.5",
      md: "h-8 px-3.5 text-[12px] gap-2",
      lg: "h-10 px-5 text-[13px] gap-2.5",
    }[size];

    const variantStyles = {
      primary:
        "bg-[#0071E3] text-white border border-[#0071E3] hover:bg-[#0077ED] hover:border-[#0077ED] active:bg-[#0062C4] shadow-sm",
      secondary:
        "bg-[#FFFFFF] text-[#1D1D1F] border border-[#D2D2D7] hover:bg-[#F5F5F7] active:bg-[#E8E8ED]",
      outline:
        "bg-transparent text-[#0071E3] border border-[#0071E3] hover:bg-[#0071E3]/5 active:bg-[#0071E3]/10",
      destructive:
        "bg-transparent text-[#D93025] border border-[#D93025] hover:bg-[#D93025] hover:text-white active:bg-[#B3261E]",
      ghost:
        "bg-transparent text-[#1D1D1F] border border-transparent hover:bg-[#F5F5F7] active:bg-[#E8E8ED]",
    }[variant];

    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        disabled={isDisabled}
        className={`${baseStyles} ${sizeStyles} ${variantStyles} ${className}`}
        {...props}
      >
        {loading ? (
          <>
            <svg
              className="animate-spin h-3.5 w-3.5 text-current shrink-0"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8H4z"
              />
            </svg>
            <span>{loadingText || children}</span>
          </>
        ) : (
          <>
            {icon && <span className="shrink-0 flex items-center">{icon}</span>}
            {children}
            {iconRight && <span className="shrink-0 flex items-center">{iconRight}</span>}
          </>
        )}
      </button>
    );
  }
);

Button.displayName = "Button";
