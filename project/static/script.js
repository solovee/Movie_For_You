const modal = document.getElementById("catalogo");
const btnAdd = document.getElementById("btnAdd");
const spanClose = document.getElementsByClassName("close")[0];
const btnEnviar = document.getElementById("btnEnviar");
const listaRecomendados = document.getElementById("lista-recomendados");

btnAdd.onclick = () => modal.style.display = "block";
spanClose.onclick = () => modal.style.display = "none";
window.onclick = (e) => { if(e.target == modal) modal.style.display = "none"; }

btnEnviar.onclick = async () => {
    const filmesSelecionados = [];
    document.querySelectorAll(".filme-item").forEach(item => {
        const checkbox = item.querySelector(".filme-checkbox");
        const nota = item.querySelector(".nota").value;
        if(checkbox.checked && nota) {
            filmesSelecionados.push({movieId: checkbox.dataset.id, nota: nota});
        }
    });

    if(filmesSelecionados.length < 7){
        alert("Selecione pelo menos 7 filmes com nota");
        return;
    }

    // Chama backend
    const res = await fetch("/recomendar", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(filmesSelecionados)
    });

    if(res.ok){
        const data = await res.json();
        listaRecomendados.innerHTML = "";
        data.recomendados.forEach(f => {
            const li = document.createElement("li");
            li.textContent = `${f.nome} (ID: ${f.movieId})`;
            listaRecomendados.appendChild(li);
        });
        modal.style.display = "none";
    } else {
        const erro = await res.json();
        alert(erro.erro);
    }
};
