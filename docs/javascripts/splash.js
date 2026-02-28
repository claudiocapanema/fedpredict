document.addEventListener("DOMContentLoaded", function () {

  const splashDuration = 2000;
  const fadeDuration = 400;
  const forceSplash = window.location.search.includes("splash=true");

  if (forceSplash || !localStorage.getItem("visitedFedPredict")) {

    const splash = document.createElement("div");
    splash.id = "splash-screen";
    splash.innerHTML = `
      <div class="splash-content">
        <h1 class="logo">FedPredict</h1>
        <div class="loader">
          <div class="progress"></div>
        </div>
      </div>
    `;

    document.body.appendChild(splash);

    setTimeout(() => {
      splash.classList.add("fade-out");
      localStorage.setItem("visitedFedPredict", "true");

      setTimeout(() => {
        splash.remove();
      }, fadeDuration);

    }, splashDuration);
  }
});

localStorage.clear();