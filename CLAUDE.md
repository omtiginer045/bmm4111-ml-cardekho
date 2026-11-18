# CLAUDE.md — bmm4111-ml-cardekho

BMM4111 Makine Öğrenmesi dönem ödevi (Balıkesir Üniversitesi). Öğrenci: Ömer Coşkun.
Ödev metni: `docs/BMM4111_Donem_Odevi_f26.pdf`. Uygulama planı: `docs/superpowers/plans/2026-11-14-cardekho-ml-odevi.md`.

## Nasıl çalışıyoruz

- Planı `superpowers:executing-plans` ile uygula. Kod yazarken `karpathy-guidelines` ilkelerine uy: en az kod, istenmeyen özellik yok, varsayımları açıkça söyle.
- Her adımda TDD uygula: önce test yaz, başarısız olduğunu gör, sonra kodu yaz.
- **Her Task sonunda:** `pytest -q` yeşil → commit → `git push origin main` → kısa Türkçe özet → **benim onayımı bekle.** Onay gelmeden sonraki Task'a geçme.
- Plandan sapman gerekirse nedenini özette açıkça yaz.

## Kurallar

- Kullanıcıyla Türkçe konuş. Kod yorumları ve README Türkçe; değişken ve fonksiyon adları İngilizce.
- Bir görevin hedefi diğer görevlerde özellik olamaz: sınıflandırma X'inde `selling_price`, regresyon X'inde `transmission` yok.
- Ön işleme yalnız sklearn `Pipeline` içinde yapılır. Test kümesi yalnız son değerlendirmede kullanılır. `SEED = 42`.
- **Raporu yazma.** Rapor el yazısıyla ve öğrencinin kendi cümleleriyle yazılacak. Sonuç tablolarındaki "Yorum" sütunları boş kalır.
- Force push, geçmişi yeniden yazma ve repo silme yok.

## Komutlar

- Test: `pytest -q`
- Notebook: `jupyter nbconvert --to notebook --execute notebooks/ML_F26_CarDekho.ipynb --output /tmp/run.ipynb`
