const queryBox = document.querySelector("#q");
const go = document.querySelector("#go");
const out = document.querySelector("#out");

function esc(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[ch]));
}

function money(value) {
  return `$${Number(value).toFixed(2)}`;
}

function chips(goal) {
  const items = [];
  items.push(`${goal.meals_requested} meals`);
  items.push(`${goal.servings} people`);
  if (goal.budget != null) items.push(`under ${money(goal.budget)}`);
  if (goal.high_protein) items.push("high protein");
  if (goal.vegan) items.push("vegan");
  else if (goal.vegetarian) items.push("vegetarian");
  if (goal.gluten_free) items.push("gluten-free");
  if (goal.dairy_free) items.push("dairy-free");
  if (goal.nut_free) items.push("nut-free");
  if (goal.shellfish_free) items.push("no shellfish");
  if (goal.soy_free) items.push("soy-free");
  if (goal.exclude_pork) items.push("no pork");
  if (goal.exclude_beef) items.push("no beef");
  return items.map((item) => `<span class="chip">${item}</span>`).join("");
}

function render(cart) {
  const meals = cart.meals.map((meal) => `
    <article class="meal${meal.kept ? "" : " dropped"}">
      <div class="tag">${meal.kept ? `${meal.minutes} min` : "dropped"}</div>
      <div>
        <strong>${esc(meal.name)}</strong>
        <p>${esc(meal.blurb)}</p>
      </div>
      <div>${meal.protein_g.toFixed(0)}g protein</div>
    </article>
  `).join("");
  const lines = cart.lines.map((line) => `
    <div class="line${line.deal_pct >= 0.2 ? " sale" : ""}">
      <div>
        <strong>${esc(line.ingredient)}</strong>
        <div class="who">${esc(line.brand)} · ${esc(line.product)} · ${line.size_g}g · ${line.packs} × ${money(line.unit_price)}</div>
        ${line.note ? `<p class="note">${esc(line.note)}</p>` : ""}
      </div>
      <div class="price">${money(line.line_total)}</div>
    </div>
  `).join("");
  const notes = (cart.notes || []).map((note) => `<p class="note">${esc(note)}</p>`).join("");
  out.innerHTML = `
    <div class="chips">${chips(cart.goal)}</div>
    <section class="meals">${meals}</section>
    <section class="receipt">
      ${lines || `<p class="pending">No products fit this request.</p>`}
      ${notes}
      <div class="totals">
        <span>Cart</span><span>${money(cart.total)}</span>
        <span>Flyer savings</span><span>${money(cart.savings)}</span>
        <span>Protein per person, per dinner</span><span>${cart.protein_each_dinner.toFixed(0)}g</span>
      </div>
    </section>
    <p class="meta">${cart.planner} planner · ${cart.backend} retrieval · ${cart.latency_ms} ms</p>
  `;
}

async function build() {
  go.disabled = true;
  out.innerHTML = `<p class="pending">Planning meals and pricing the cart…</p>`;
  try {
    const response = await fetch("/api/cart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: queryBox.value }),
    });
    const body = await response.json();
    if (!response.ok) {
      out.innerHTML = `<p class="error">${body.detail || "Could not build a cart."}</p>`;
      return;
    }
    render(body);
  } catch (error) {
    out.innerHTML = `<p class="error">The cart service did not respond.</p>`;
  } finally {
    go.disabled = false;
  }
}

go.addEventListener("click", build);
document.querySelectorAll("#examples button").forEach((button) => {
  button.addEventListener("click", () => {
    queryBox.value = button.textContent;
    build();
  });
});
