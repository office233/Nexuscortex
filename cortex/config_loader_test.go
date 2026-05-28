package cortex

import (
	"os"
	"path/filepath"
	"testing"
)

// TestLoadConfig_EmptyPath garantează că LoadConfig("") returnează exact
// DefaultConfig() fără să atingă disk-ul.
func TestLoadConfig_EmptyPath(t *testing.T) {
	cfg, err := LoadConfig("")
	if err != nil {
		t.Fatalf("LoadConfig(\"\") returned err: %v", err)
	}
	want := DefaultConfig()
	if cfg.Seed != want.Seed || cfg.DataDir != want.DataDir {
		t.Fatalf("LoadConfig(\"\") did not return DefaultConfig: seed=%d datadir=%s",
			cfg.Seed, cfg.DataDir)
	}
}

// TestLoadConfig_MissingFile verifică că lipsa fișierului produce
// eroare clară (nu fallback silent).
func TestLoadConfig_MissingFile(t *testing.T) {
	_, err := LoadConfig(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err == nil {
		t.Fatal("LoadConfig pe path inexistent ar trebui să eșueze")
	}
}

// TestLoadConfig_PartialOverride verifică modelul de merge: JSON-ul
// suprascrie doar câmpurile prezente; restul rămân din DefaultConfig.
func TestLoadConfig_PartialOverride(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	// JSON care suprascrie DOAR seed-ul și un hyperparametru Adam.
	content := `{
		"seed": 1234,
		"adam_beta1": 0.95
	}`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig partial: %v", err)
	}
	if cfg.Seed != 1234 {
		t.Errorf("Seed: vrut 1234, primit %d", cfg.Seed)
	}
	if cfg.AdamBeta1 != 0.95 {
		t.Errorf("AdamBeta1: vrut 0.95, primit %f", cfg.AdamBeta1)
	}
	// Câmpurile neatinse trebuie să rămână la default.
	def := DefaultConfig()
	if cfg.DataDir != def.DataDir {
		t.Errorf("DataDir nu trebuia atins: vrut %q, primit %q", def.DataDir, cfg.DataDir)
	}
	if cfg.AdamBeta2 != def.AdamBeta2 {
		t.Errorf("AdamBeta2 nu trebuia atins: vrut %f, primit %f", def.AdamBeta2, cfg.AdamBeta2)
	}
	if cfg.TransformerEmbedDim != def.TransformerEmbedDim {
		t.Errorf("TransformerEmbedDim nu trebuia atins")
	}
}

// TestLoadConfig_InvalidJSON verifică că JSON-ul corupt produce eroare,
// nu panic.
func TestLoadConfig_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte("{not valid json"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := LoadConfig(path)
	if err == nil {
		t.Fatal("LoadConfig pe JSON corupt ar trebui să eșueze")
	}
}

// TestLoadConfig_EmptyFile verifică că un fișier gol e respins (operatorul
// probabil a uitat să-l populeze; merge silent ar fi periculos).
func TestLoadConfig_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.json")
	if err := os.WriteFile(path, []byte(""), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := LoadConfig(path)
	if err == nil {
		t.Fatal("LoadConfig pe fișier gol ar trebui să eșueze")
	}
}

// TestSaveLoadRoundtrip salvează DefaultConfig() și-l reîncarcă;
// rezultatul trebuie să fie identic pe câmpurile cheie.
func TestSaveLoadRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.json")
	orig := DefaultConfig()
	if err := SaveConfig(path, orig); err != nil {
		t.Fatalf("SaveConfig: %v", err)
	}
	loaded, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig roundtrip: %v", err)
	}
	if loaded.Seed != orig.Seed {
		t.Errorf("roundtrip Seed: vrut %d, primit %d", orig.Seed, loaded.Seed)
	}
	if loaded.AdamBeta1 != orig.AdamBeta1 {
		t.Errorf("roundtrip AdamBeta1: vrut %f, primit %f", orig.AdamBeta1, loaded.AdamBeta1)
	}
	if loaded.BrocaDecodeTopK != orig.BrocaDecodeTopK {
		t.Errorf("roundtrip BrocaDecodeTopK: vrut %d, primit %d",
			orig.BrocaDecodeTopK, loaded.BrocaDecodeTopK)
	}
}

// TestResolveConfigPath_Precedence: flag > env > auto-detect > "".
func TestResolveConfigPath_Precedence(t *testing.T) {
	// 1. Flag explicit câștigă mereu, chiar dacă fișierul nu există.
	p, src := ResolveConfigPath("explicit.json")
	if p != "explicit.json" || src != "flag" {
		t.Errorf("flag precedence: vrut (explicit.json,flag), primit (%s,%s)", p, src)
	}
	// 2. Env variable câștigă în absența flag-ului.
	t.Setenv(ConfigEnvVar, "from-env.json")
	p, src = ResolveConfigPath("")
	if p != "from-env.json" || src != "env" {
		t.Errorf("env precedence: vrut (from-env.json,env), primit (%s,%s)", p, src)
	}
	// 3. Fără flag și fără env, dacă nu există fișier implicit → "".
	t.Setenv(ConfigEnvVar, "")
	// Mută working dir într-un temp ca să nu prindem un fișier existent.
	tmp := t.TempDir()
	oldWd, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(oldWd) })
	if err := os.Chdir(tmp); err != nil {
		t.Fatal(err)
	}
	p, src = ResolveConfigPath("")
	if p != "" || src != "" {
		t.Errorf("auto-detect lipsă: vrut (\"\",\"\"), primit (%s,%s)", p, src)
	}
}
