# Download priority reference papers for AIC2026
# Run from docs\papers\ directory
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# 0. Patch: ensure TLS 1.2
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$papers = @(
  # Foundation VLMs and dual encoders
  @{url="https://arxiv.org/pdf/2502.14786"; out="foundation-vlm/SigLIP-2_arxiv-2502.14786.pdf"},
  @{url="https://arxiv.org/pdf/2507.22062"; out="foundation-vlm/MetaCLIP-2_arxiv-2507.22062.pdf"},
  @{url="https://arxiv.org/pdf/2303.15389"; out="foundation-vlm/EVA-CLIP_arxiv-2303.15389.pdf"},
  @{url="https://arxiv.org/pdf/2103.00020"; out="foundation-vlm/CLIP_arxiv-2103.00020.pdf"},
  @{url="https://arxiv.org/pdf/2403.15378"; out="foundation-vlm/Long-CLIP_arxiv-2403.15378.pdf"},
  @{url="https://arxiv.org/pdf/2403.17007"; out="foundation-vlm/DreamLIP_arxiv-2403.17007.pdf"},
  @{url="https://arxiv.org/pdf/2301.12597"; out="foundation-vlm/BLIP-2_arxiv-2301.12597.pdf"},
  @{url="https://arxiv.org/pdf/2208.10442"; out="foundation-vlm/BEiT-3_arxiv-2208.10442.pdf"},
  @{url="https://arxiv.org/pdf/2106.11097"; out="foundation-vlm/CLIP2Video_arxiv-2106.11097.pdf"},
  @{url="https://arxiv.org/pdf/2207.14757"; out="foundation-vlm/ALADIN_arxiv-2207.14757.pdf"},

  # VLM/MLLM
  @{url="https://arxiv.org/pdf/2502.13923"; out="foundation-vlm/Qwen2.5-VL_arxiv-2502.13923.pdf"},
  @{url="https://arxiv.org/pdf/2508.18265"; out="foundation-vlm/InternVL-3.5_arxiv-2508.18265.pdf"},
  @{url="https://arxiv.org/pdf/2410.07073"; out="foundation-vlm/Pixtral-12B_arxiv-2410.07073.pdf"},

  # Video understanding
  @{url="https://arxiv.org/pdf/2403.15377"; out="foundation-vlm/InternVideo2_arxiv-2403.15377.pdf"},
  @{url="https://arxiv.org/pdf/2506.09985"; out="foundation-vlm/V-JEPA-2_arxiv-2506.09985.pdf"},
  @{url="https://arxiv.org/pdf/2310.01852"; out="foundation-vlm/LanguageBind_arxiv-2310.01852.pdf"},

  # Text retrievers
  @{url="https://arxiv.org/pdf/2402.03216"; out="foundation-vlm/BGE-M3_arxiv-2402.03216.pdf"},
  @{url="https://arxiv.org/pdf/2409.10173"; out="foundation-vlm/Jina-Embeddings-v3_arxiv-2409.10173.pdf"},
  @{url="https://arxiv.org/pdf/2407.19669"; out="foundation-vlm/mGTE_arxiv-2407.19669.pdf"},
  @{url="https://arxiv.org/pdf/2403.06789"; out="foundation-vlm/SPLADE-v3_arxiv-2403.06789.pdf"},

  # Vietnamese
  @{url="https://arxiv.org/pdf/2406.02555"; out="vietnamese-multimodal/PhoWhisper_arxiv-2406.02555.pdf"},
  @{url="https://arxiv.org/pdf/2507.05595"; out="vietnamese-multimodal/PaddleOCR-3.0_arxiv-2507.05595.pdf"},

  # ColPali / Visual document retrieval
  @{url="https://arxiv.org/pdf/2407.01449"; out="foundation-vlm/ColPali_arxiv-2407.01449.pdf"},
  @{url="https://arxiv.org/pdf/2410.10594"; out="foundation-vlm/VisRAG_arxiv-2410.10594.pdf"},

  # LSC and VBS
  @{url="https://arxiv.org/pdf/2506.06743"; out="lsc-systems/LSC-SOTA-review_arxiv-2506.06743.pdf"},
  @{url="https://arxiv.org/pdf/2502.15683"; out="vbs-systems/VBS-2024-results_arxiv-2502.15683.pdf"},
  @{url="https://arxiv.org/pdf/2509.12000"; out="vbs-systems/VBS-2025-results_arxiv-2509.12000.pdf"},
  @{url="https://arxiv.org/pdf/2503.17116"; out="benchmarks/CASTLE-2024_arxiv-2503.17116.pdf"},

  # Datasets & benchmarks
  @{url="https://arxiv.org/pdf/2504.02060"; out="benchmarks/LSC-ADL_arxiv-2504.02060.pdf"},
  @{url="https://arxiv.org/pdf/2306.01069"; out="benchmarks/TimelineQA_arxiv-2306.01069.pdf"},

  # Agentic retrieval
  @{url="https://arxiv.org/pdf/2507.13374"; out="agentic-retrieval/SmartRouting_arxiv-2507.13374.pdf"},

  # HCMC AIC prior art
  @{url="https://arxiv.org/pdf/2512.06334"; out="lsc-systems/AIC2025-EEIoT-newbie_arxiv-2512.06334.pdf"},
  @{url="https://arxiv.org/pdf/2605.16120"; out="lsc-systems/MERVIN-AIC2025_arxiv-2605.16120.pdf"},
  @{url="https://arxiv.org/pdf/2512.13169"; out="lsc-systems/QUEST-DANTE-AIC2025_arxiv-2512.13169.pdf"},
  @{url="https://arxiv.org/pdf/2512.12935"; out="agentic-retrieval/CascadedMM-Agent_arxiv-2512.12935.pdf"},

  # Egocentric/multimodal datasets
  @{url="https://arxiv.org/pdf/2204.04405"; out="benchmarks/EPIC-Kitchens-100_arxiv-2204.04405.pdf"},
  @{url="https://arxiv.org/pdf/2110.07058"; out="benchmarks/Ego4D_arxiv-2110.07058.pdf"}
)

$ok = 0; $fail = 0
foreach ($p in $papers) {
  if (Test-Path $p.out) { Write-Host "skip (exists): $($p.out)"; continue }
  Write-Host "downloading: $($p.url) -> $($p.out)"
  try {
    Invoke-WebRequest -Uri $p.url -OutFile $p.out -UserAgent "Mozilla/5.0" -TimeoutSec 60 -ErrorAction Stop
    $ok++
  } catch {
    Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Yellow
    $fail++
  }
}
Write-Host ""
Write-Host ("DONE. ok={0} fail={1}" -f $ok, $fail)
