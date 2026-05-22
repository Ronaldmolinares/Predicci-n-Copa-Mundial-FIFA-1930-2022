document.addEventListener("DOMContentLoaded", () => {

    // Configuración de API local
    const API_BASE = "http://127.0.0.1:8000/api/v1";

    // Equipos disponibles del dataset
    const TEAMS = [
        "Algeria", "Angola", "Argentina", "Australia", "Austria", "Belgium", "Bolivia",
        "Bosnia and Herzegovina", "Brazil", "Bulgaria", "Cameroon", "Canada", "Chile",
        "China PR", "Colombia", "Costa Rica", "Croatia", "Cuba", "Czech Republic",
        "Czechoslovakia", "Côte d'Ivoire", "Denmark", "Ecuador", "Egypt", "England",
        "FR Yugoslavia", "France", "Germany", "Germany DR", "Ghana", "Greece", "Haiti",
        "Honduras", "Hungary", "IR Iran", "Iceland", "Iraq", "Italy", "Jamaica", "Japan",
        "Korea DPR", "Korea Republic", "Mexico", "Morocco", "Netherlands", "New Zealand",
        "Nigeria", "Northern Ireland", "Norway", "Panama", "Paraguay", "Peru", "Poland",
        "Portugal", "Qatar", "Republic of Ireland", "Romania", "Russia", "Saudi Arabia",
        "Scotland", "Senegal", "Serbia", "Serbia and Montenegro", "Slovakia", "Slovenia",
        "South Africa", "Soviet Union", "Spain", "Sweden", "Switzerland", "Togo",
        "Trinidad and Tobago", "Tunisia", "Türkiye", "Ukraine", "United Arab Emirates",
        "United States", "Uruguay", "Wales", "West Germany", "Yugoslavia", "Zaire"
    ];

    // Fases estándar unificadas (en orden cronológico)
    // Mapeo realizado desde 12 etiquetas originales a 6 estándar
    const STAGES = [
        "Group Stage",      // Absorbe: First group stage, Second group stage, Group stage play-off, Group stage
        "Round of 16",      // Absorbe: First round, Second round, Round of 16
        "Quarter-finals",   // Se mantiene
        "Semi-finals",      // Se mantiene
        "Third-place match", // Se mantiene
        "Final"             // Absorbe: Final stage, Final
    ];

    // Inicializar dropdowns con datos
    function initializeDropdowns() {
        const homeTeamSelect = document.getElementById("home-team");
        const awayTeamSelect = document.getElementById("away-team");
        const clusterTeamSelect = document.getElementById("cluster-team");
        const stageSelect = document.getElementById("stage");

        // Llenar equipos en ambos dropdowns
        TEAMS.forEach(team => {
            const option = document.createElement("option");
            option.value = team;
            option.textContent = team;
            homeTeamSelect.appendChild(option);

            const cloneOption = document.createElement("option");
            cloneOption.value = team;
            cloneOption.textContent = team;
            awayTeamSelect.appendChild(cloneOption);

            const clusterOption = document.createElement("option");
            clusterOption.value = team;
            clusterOption.textContent = team;
            clusterTeamSelect.appendChild(clusterOption);
        });

        // Llenar fases
        STAGES.forEach(stage => {
            const option = document.createElement("option");
            option.value = stage;
            option.textContent = stage;
            stageSelect.appendChild(option);
        });

        // Establecer valores por defecto
        homeTeamSelect.value = "Argentina";
        awayTeamSelect.value = "France";
        clusterTeamSelect.value = "Colombia";
        stageSelect.value = "Final";
    }

    // Validación de año (máximo 4 dígitos)
    const yearInputs = document.querySelectorAll('input[type="number"][id*="year"]');
    yearInputs.forEach(input => {
        input.addEventListener("input", (e) => {
            if (e.target.value.length > 4) {
                e.target.value = e.target.value.slice(0, 4);
            }
        });
    });

    // Inicializar dropdowns al cargar
    initializeDropdowns();

    // Navegación de Pestañas
    const tabPredict = document.getElementById("tab-predict");
    const tabCluster = document.getElementById("tab-cluster");
    const panelPredict = document.getElementById("panel-predict");
    const panelCluster = document.getElementById("panel-cluster");

    function switchTab(activeTab, activePanel, inactiveTab, inactivePanel) {
        activeTab.classList.add("active");
        inactiveTab.classList.remove("active");

        inactivePanel.classList.remove("active");
        setTimeout(() => {
            activePanel.classList.add("active");
        }, 50); // Pequeño delay para la animación
    }

    tabPredict.addEventListener("click", () => switchTab(tabPredict, panelPredict, tabCluster, panelCluster));
    tabCluster.addEventListener("click", () => switchTab(tabCluster, panelCluster, tabPredict, panelPredict));

    // Toast de Error
    const toast = document.getElementById("error-toast");
    const errorMsg = document.getElementById("error-msg");
    function showError(message) {
        errorMsg.textContent = message;
        toast.classList.remove("hidden");
        setTimeout(() => toast.classList.add("hidden"), 5000);
    }

    // Funciones Helper de UI
    function setLoading(btn, isLoading) {
        const span = btn.querySelector("span");
        const icon = btn.querySelector("i");
        if (isLoading) {
            btn.disabled = true;
            icon.className = "fa-solid fa-spinner fa-spin";
        } else {
            btn.disabled = false;
            // Restaurar icono original
            icon.className = btn.id === "btn-predict" ? "fa-solid fa-wand-magic-sparkles" : "fa-solid fa-microchip";
        }
    }

    // ==========================================
    // LÓGICA DE PREDICCIÓN DE PARTIDOS
    // ==========================================
    const formPredict = document.getElementById("form-predict");
    const btnPredict = document.getElementById("btn-predict");
    const resultPredict = document.getElementById("result-predict");

    formPredict.addEventListener("submit", async (e) => {
        e.preventDefault();

        const payload = {
            home_team: document.getElementById("home-team").value.trim(),
            away_team: document.getElementById("away-team").value.trim(),
            year: parseInt(document.getElementById("year").value),
            stage: document.getElementById("stage").value.trim()
        };

        setLoading(btnPredict, true);
        resultPredict.classList.add("hidden");

        try {
            const response = await fetch(`${API_BASE}/predict/match`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Error en el servidor");
            }

            renderPrediction(data);
        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(btnPredict, false);
        }
    });

    function renderPrediction(data) {
        // Mapeo estético del ganador
        const winnerEl = document.getElementById("predicted-winner");
        const pc = data.predicted_class;
        if (pc === "home_win") winnerEl.textContent = "Victoria Local";
        else if (pc === "draw") winnerEl.textContent = "Empate";
        else if (pc === "away_win") winnerEl.textContent = "Victoria Visitante";
        else winnerEl.textContent = pc;

        // Probabilidades (asume que la API devuelve llaves: home_win, draw, away_win)
        const pHome = (data.probabilities["home_win"] || 0) * 100;
        const pDraw = (data.probabilities["draw"] || 0) * 100;
        const pAway = (data.probabilities["away_win"] || 0) * 100;

        // Animar barras de progreso tras renderizar contenedor
        resultPredict.classList.remove("hidden");

        // Reset a 0 para que la animación se dispare de nuevo
        document.getElementById("bar-home").style.width = "0%";
        document.getElementById("bar-draw").style.width = "0%";
        document.getElementById("bar-away").style.width = "0%";

        setTimeout(() => {
            document.getElementById("bar-home").style.width = `${pHome}%`;
            document.getElementById("val-home").textContent = `${pHome.toFixed(1)}%`;

            document.getElementById("bar-draw").style.width = `${pDraw}%`;
            document.getElementById("val-draw").textContent = `${pDraw.toFixed(1)}%`;

            document.getElementById("bar-away").style.width = `${pAway}%`;
            document.getElementById("val-away").textContent = `${pAway.toFixed(1)}%`;
        }, 100);
    }

    // ==========================================
    // LÓGICA DE CLUSTERING DE EQUIPOS
    // ==========================================
    const formCluster = document.getElementById("form-cluster");
    const btnCluster = document.getElementById("btn-cluster");
    const resultCluster = document.getElementById("result-cluster");

    formCluster.addEventListener("submit", async (e) => {
        e.preventDefault();

        const payload = {
            team: document.getElementById("cluster-team").value.trim(),
            year: parseInt(document.getElementById("cluster-year").value)
        };

        setLoading(btnCluster, true);
        resultCluster.classList.add("hidden");

        try {
            const response = await fetch(`${API_BASE}/cluster/team`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Error en el servidor");
            }

            renderCluster(data);
        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(btnCluster, false);
        }
    });

    function renderCluster(data) {
        const badge = document.getElementById("cluster-badge");
        badge.textContent = data.cluster_label;
        badge.className = `cluster-badge cluster-${data.cluster_id}`;

        resultCluster.classList.remove("hidden");
    }

});
