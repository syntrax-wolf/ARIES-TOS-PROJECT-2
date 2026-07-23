import { useEffect, useState, type FormEvent } from "react";
import { motion } from "framer-motion";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { Logomark } from "../components/StepHeader";
import { ApiError, cacheStudent, readCachedStudent, register } from "../lib/api";
import { useSylvaStore } from "../state/store";

export function Register() {
  const setStudent = useSylvaStore((s) => s.setStudent);
  const setPage = useSylvaStore((s) => s.setPage);

  const [name, setName] = useState("");
  const [entryNumber, setEntryNumber] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Pre-fill from the last session so a refresh isn't a chore. They still
  // confirm, so the server stays the source of truth for the student id.
  useEffect(() => {
    const cached = readCachedStudent();
    if (cached) {
      setName(cached.name);
      setEntryNumber(cached.entry_number);
    }
  }, []);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);

    const trimmedName = name.trim();
    const trimmedEntry = entryNumber.trim().toUpperCase();
    if (!trimmedName) return setError("Please enter your full name.");
    if (!trimmedEntry) return setError("Please enter your entry number.");

    setBusy(true);
    try {
      const student = await register(trimmedName, trimmedEntry);
      cacheStudent(student);
      setStudent(student);
      setPage("select");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex items-center gap-2 px-8 py-6 text-brand-dark sm:px-12">
        <Logomark className="h-6 w-6" />
        <span className="font-serif text-lg font-semibold tracking-tight">Sylva</span>
      </header>

      <main className="mx-auto flex w-full max-w-lg flex-1 flex-col justify-center px-6 pb-24 sm:px-8">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        >
          <div className="mb-10 text-center">
            <h1 className="text-4xl text-ink sm:text-[42px]">Before you begin</h1>
            <p className="mx-auto mt-3 max-w-md text-[17px] leading-relaxed text-ink-soft">
              Your runs are scored on the server and ranked against everyone else's, so we need to
              know who's training.
            </p>
          </div>

          <Card className="p-7">
            <form onSubmit={handleSubmit} noValidate>
              <label htmlFor="name" className="block text-sm font-medium text-ink">
                Your name
              </label>
              <input
                id="name"
                type="text"
                value={name}
                maxLength={60}
                autoComplete="name"
                placeholder="Ada Lovelace"
                onChange={(e) => setName(e.target.value)}
                className="mt-1.5 w-full rounded-xl border border-border bg-surface px-3.5 py-2.5 text-[15px] text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none"
              />

              <label htmlFor="entry" className="mt-5 block text-sm font-medium text-ink">
                Entry number
              </label>
              <input
                id="entry"
                type="text"
                value={entryNumber}
                maxLength={20}
                autoComplete="off"
                spellCheck={false}
                placeholder="2023CS10123"
                onChange={(e) => setEntryNumber(e.target.value.toUpperCase())}
                className="mt-1.5 w-full rounded-xl border border-border bg-surface px-3.5 py-2.5 font-mono text-[15px] uppercase tracking-wide text-ink placeholder:font-sans placeholder:normal-case placeholder:tracking-normal placeholder:text-ink-faint focus:border-brand focus:outline-none"
              />
              <p className="mt-1.5 text-[13px] text-ink-faint">
                Used to keep your place on the leaderboard across sessions.
              </p>

              {error && (
                <motion.p
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-5 rounded-xl bg-amber-soft px-3.5 py-2.5 text-sm text-danger"
                >
                  {error}
                </motion.p>
              )}

              <Button type="submit" size="lg" disabled={busy} className="mt-6 w-full">
                {busy ? "Checking you in…" : "Start training"}
              </Button>
            </form>
          </Card>
        </motion.div>
      </main>
    </div>
  );
}
