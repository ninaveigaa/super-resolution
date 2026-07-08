import numpy as np
import matplotlib.pyplot as plt
 
# Tenta importar métricas de qualidade de imagem (opcional)
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
 
 
def compute_metrics(true_img: np.ndarray, pred_img: np.ndarray) -> dict:
    """Calcula métricas de erro entre duas imagens/arrays 2D."""
    diff = np.abs(true_img - pred_img)
 
    mae = np.mean(diff)
    mse = np.mean((true_img - pred_img) ** 2)
    rmse = np.sqrt(mse)
 
    # PSNR: precisa do range de valores da imagem "true"
    data_range = true_img.max() - true_img.min()
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
 
    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "PSNR": psnr}
 
    if HAS_SSIM:
        metrics["SSIM"] = ssim(
            true_img, pred_img, data_range=data_range
        )
 
    return metrics
 
 
def plot_comparison(
    true_img: np.ndarray,
    pred_img: np.ndarray,
    cmap: str = "gray",
    save_path: str = None,
):
    """
    Plota True, Pred e a diferença (resíduo) lado a lado (3 painéis).
 
    - True e Pred usam escala de cinza (grayscale), igual à imagem original,
      com a MESMA escala de valores (vmin/vmax compartilhados) entre os dois.
    - Difference mostra o resíduo pixel a pixel (|True - Pred|), também em
      escala de cinza, mas com sua própria escala (0 até o valor máximo do
      resíduo), já que os valores de erro são bem menores que os da imagem
      original.
    - Sem título, sem eixos, sem borda e sem barra de legenda — só as imagens.
    """
    diff = np.abs(true_img - pred_img)
 
    # Escala compartilhada entre True e Pred
    vmin = min(true_img.min(), pred_img.min())
    vmax = max(true_img.max(), pred_img.max())
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
 
    axes[0].imshow(true_img, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].imshow(pred_img, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].imshow(diff, cmap=cmap, vmin=0, vmax=diff.max())
 
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
 
    plt.subplots_adjust(wspace=0.02)
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura salva em: {save_path}")
 
    plt.show()
 
    # Imprime métricas no console
    metrics = compute_metrics(true_img, pred_img)
    print("\n--- Métricas de erro ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
 
 
def load_image_as_array(path: str) -> np.ndarray:
    """Carrega uma imagem de arquivo (png/jpg/etc) como array 2D (grayscale)."""
    from PIL import Image
    img = Image.open(path).convert("L")  # converte pra escala de cinza
    return np.array(img, dtype=np.float32)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # EXEMPLO: substitua isso pelo carregamento dos seus próprios dados
    # ------------------------------------------------------------------

    # Opção A: carregar de arquivos de imagem
    true_img = load_image_as_array("/Users/ninaveiga/Documents/PROJETOS/Estágio/Imagens/HR_EDSR_v1.png")
    pred_img = load_image_as_array("/Users/ninaveiga/Documents/PROJETOS/Estágio/Imagens/SR_EDSR_v1.png")


    plot_comparison(true_img, pred_img, save_path="comparison_output.png")