/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_AIC_SHARED_SECRET?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
