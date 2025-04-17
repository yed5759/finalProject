import * as React from "react"

const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type, placeholder, value, onChange, dir, ...props }, ref) => {
    return (
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        dir={dir} // תומך ב-dir
        className={
          "flex h-10 w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-950 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 " + 
          (className || "")
        }
        ref={ref}
        {...props} // כל המאפיינים הנוספים
      />
    );
  }
);

Input.displayName = "Input";

export default Input;
export { Input };