// Builds the left-hand timeline rail from the sections in the DOM (single source
// of truth = the HTML itself) and highlights the section currently in view.

function initNav() {
  const rail = document.querySelector(".rail-list");
  const sections = Array.from(document.querySelectorAll("section.mech"));
  const links = [];

  // Era headings in the rail, so the field-priority arc is visible while
  // scrolling rather than only stated once in the hero.
  const ERA_NAMES = {
    "make-it-work": "Make it work",
    "bills-come-due": "The bills come due",
    "position-rebuilt": "Position rebuilt",
    "stretch-context": "Stretch the context",
    "state-returns": "State + sparsity return",
    compression: "Compression gets aggressive",
    now: "Where this leaves us",
  };
  let lastEra = null;

  sections.forEach((sec) => {
    const li = document.createElement("li");
    const era = sec.dataset.era;
    if (era && era !== lastEra) {
      lastEra = era;
      const head = document.createElement("div");
      head.className = "rail-era";
      head.textContent = ERA_NAMES[era] || era;
      li.appendChild(head);
    }
    const a = document.createElement("a");
    a.href = "#" + sec.id;
    a.className = "rail-link";
    const dateSpan = document.createElement("span");
    dateSpan.className = "rail-date";
    dateSpan.textContent = sec.dataset.date || "";
    a.append(sec.dataset.short || sec.querySelector("h2").textContent, dateSpan);
    li.appendChild(a);
    rail.appendChild(li);
    links.push({ id: sec.id, a });
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const link = links.find((l) => l.id === entry.target.id);
        if (!link) return;
        if (entry.isIntersecting) {
          links.forEach((l) => l.a.classList.remove("active"));
          link.a.classList.add("active");
        }
      });
    },
    { rootMargin: "-10% 0px -70% 0px", threshold: 0 }
  );

  sections.forEach((sec) => observer.observe(sec));
}
