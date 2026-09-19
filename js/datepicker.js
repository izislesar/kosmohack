(function () {

    const MONTHS = [
        "Январь", "Февраль", "Март", "Апрель",
        "Май", "Июнь", "Июль", "Август",
        "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"
    ];

    let popover = null;
    let grid = null;
    let titleButton = null;
    let activeInput = null;
    let viewYear = 0;
    let viewMonth = 0;

    function pad2(n) { return String(n).padStart(2, "0"); }

    function toIso(y, m, d) { return `${y}-${pad2(m + 1)}-${pad2(d)}`; }

    function fromIso(iso) {
        const x = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || "");
        return x ? { y: +x[1], m: +x[2] - 1, d: +x[3] } : null;
    }

    function display(iso) {
        const p = fromIso(iso);
        return p ? `${pad2(p.d)}.${pad2(p.m + 1)}.${p.y}` : "";
    }

    function today() {
        const t = new Date();
        return { y: t.getFullYear(), m: t.getMonth(), d: t.getDate() };
    }

    function same(a, b) {
        return a && b && a.y === b.y && a.m === b.m && a.d === b.d;
    }

    function cmp(a, b) {
        return a.y - b.y || a.m - b.m || a.d - b.d;
    }

    function buildGrid(y, m) {
        const first = (new Date(y, m, 1).getDay() + 6) % 7;
        const dim = new Date(y, m + 1, 0).getDate();
        const cells = [];

        const pm = m === 0 ? 11 : m - 1;
        const py = m === 0 ? y - 1 : y;
        const dpm = new Date(py, pm + 1, 0).getDate();

        for (let i = first - 1; i >= 0; i--) {
            cells.push({ y: py, m: pm, d: dpm - i, out: true });
        }

        for (let d = 1; d <= dim; d++) {
            cells.push({ y, m, d, out: false });
        }

        const nm = m === 11 ? 0 : m + 1;
        const ny = m === 11 ? y + 1 : y;
        let d = 1;

        while (cells.length < 42) {
            cells.push({ y: ny, m: nm, d: d++, out: true });
        }

        return cells;
    }

    function getRange() {
        return {
            from: fromIso(document.getElementById("dateFrom")?.dataset.value),
            to: fromIso(document.getElementById("dateTo")?.dataset.value)
        };
    }

    function render() {
        if (!popover || !grid) return;

        if (titleButton) {
            titleButton.textContent = `${MONTHS[viewMonth]} ${viewYear}`;
        }

        const r = getRange();
        const t = today();

        grid.innerHTML = "";

        buildGrid(viewYear, viewMonth).forEach(function (c) {

            const b = document.createElement("button");
            b.type = "button";
            b.className = "datepicker-day";
            b.textContent = c.d;

            if (c.out) b.classList.add("is-outside");
            if (same(c, t)) b.classList.add("is-today");

            if (r.from && same(c, r.from)) b.classList.add("is-selected");
            if (r.to && same(c, r.to)) b.classList.add("is-selected");

            if (r.from && r.to && cmp(c, r.from) > 0 && cmp(c, r.to) < 0) {
                b.classList.add("is-in-range");
            }

            if (activeInput === "from" && r.to && cmp(c, r.to) > 0) b.disabled = true;
            if (activeInput === "to" && r.from && cmp(c, r.from) < 0) b.disabled = true;

            b.addEventListener("click", function () { select(c); });

            grid.appendChild(b);
        });
    }

    function select(c) {
        if (!activeInput) return;

        const iso = toIso(c.y, c.m, c.d);
        const inp = document.getElementById(
            activeInput === "from" ? "dateFrom" : "dateTo"
        );

        if (!inp) return;

        inp.dataset.value = iso;
        inp.value = display(iso);
        inp.dispatchEvent(new Event("change", { bubbles: true }));

        const r = getRange();

        if (r.from && r.to && cmp(r.from, r.to) > 0) {
            const to = document.getElementById("dateTo");
            if (to) {
                to.dataset.value = iso;
                to.value = display(iso);
                to.dispatchEvent(new Event("change", { bubbles: true }));
            }
        }

        hide();
    }

    function prev() {
        viewMonth--;
        if (viewMonth < 0) { viewMonth = 11; viewYear--; }
        render();
    }

    function next() {
        viewMonth++;
        if (viewMonth > 11) { viewMonth = 0; viewYear++; }
        render();
    }

    function goToday() {
        const t = today();
        viewYear = t.y;
        viewMonth = t.m;
        render();
    }

    function position(inp) {
        const rect = inp.getBoundingClientRect();
        const pw = popover.offsetWidth || 268;
        const ph = popover.offsetHeight || 320;
        const m = 8;

        let top = rect.bottom + m;
        let left = rect.left;

        if (top + ph > window.innerHeight - m) top = rect.top - ph - m;
        if (left + pw > window.innerWidth - m) left = rect.right - pw;
        if (left < m) left = m;
        if (top < m) top = m;

        popover.style.top = top + "px";
        popover.style.left = left + "px";
    }

    function show(inp) {
        activeInput = inp.dataset.datepicker || "from";

        const p = fromIso(inp.dataset.value);

        if (p) {
            viewYear = p.y;
            viewMonth = p.m;
        } else {
            const t = today();
            viewYear = t.y;
            viewMonth = t.m;
        }

        popover.hidden = false;
        render();
        position(inp);

        requestAnimationFrame(function () { position(inp); });
    }

    function hide() {
        popover.hidden = true;
        activeInput = null;
    }

    function bind(inp) {
        if (!inp) return;

        inp.addEventListener("click", function (e) {
            e.stopPropagation();

            if (popover.hidden || activeInput !== inp.dataset.datepicker) {
                show(inp);
            } else {
                hide();
            }
        });

        inp.addEventListener("focus", function () { inp.blur(); });
    }

    function init() {
        popover = document.getElementById("datepickerPopover");
        grid = document.getElementById("datepickerGrid");
        titleButton = popover?.querySelector(".datepicker-title");

        if (!popover || !grid) return;

        popover.addEventListener("click", function (e) {
            const t = e.target.closest("[data-action]");
            if (!t) return;

            if (t.dataset.action === "prev") prev();
            else if (t.dataset.action === "next") next();
            else if (t.dataset.action === "today") goToday();
        });

        bind(document.getElementById("dateFrom"));
        bind(document.getElementById("dateTo"));

        document.addEventListener("click", function (e) {
            if (popover.hidden) return;
            if (popover.contains(e.target)) return;
            if (e.target.closest("[data-datepicker]")) return;
            hide();
        });

        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape" && !popover.hidden) hide();
        });

        window.addEventListener("scroll", function () {
            if (!popover.hidden && activeInput) {
                const inp = document.getElementById(
                    activeInput === "from" ? "dateFrom" : "dateTo"
                );
                if (inp) position(inp);
            }
        }, true);

        window.addEventListener("resize", function () {
            if (!popover.hidden && activeInput) {
                const inp = document.getElementById(
                    activeInput === "from" ? "dateFrom" : "dateTo"
                );
                if (inp) position(inp);
            }
        });
    }

    document.addEventListener("DOMContentLoaded", init);

})();