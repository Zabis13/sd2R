# sdR TODO

## Архитектура

**Подход 2: Rcpp обёртки вокруг C++ stable-diffusion.cpp**

- C++ (src/sd/): токенизаторы, энкодеры, денойзер, семплер, VAE, загрузка моделей
- R: параметризация, хелперы, тестирование, высокоуровневый API
- ggmlR: libggml.a + ggml хедеры через LinkingTo

## Сделано

- [x] C++ исходники stable-diffusion.cpp скопированы в src/sd/
- [x] src/sdR_interface.cpp — Rcpp обёртки (XPtr с кастомным finalizer для sd_ctx_t, upscaler_ctx_t)
- [x] src/Makevars — сборка sd/*.cpp, линковка с libggml.a, -DGGML_MAX_NAME=128
- [x] R/pipeline.R — sd_ctx(), sd_txt2img(), sd_img2img()
- [x] R/zzz.R — константы (0-based, совпадают с C++ enum), .onLoad → sd_init_log()
- [x] R/utils.R — sd_system_info() через C++
- [x] R/image_utils.R — sd_save_image(), sd_tensor_to_image(), sd_image_to_tensor()
- [x] R/sdR-package.R — useDynLib, importFrom Rcpp
- [x] DESCRIPTION — Rcpp, LinkingTo: Rcpp + ggmlR
- [x] Удалены 18 R файлов (чистые R реализации моделей/слоёв)
- [x] Добавлены r_ggml_compat.h и ggml-vulkan.h в ggmlR inst/include, ggmlR переустановлен
- [x] Makevars: использует installed ggmlR через LinkingTo, -include r_ggml_compat.h
- [x] Компиляция и установка sdR — OK
- [x] NAMESPACE обновлён через roxygen2
- [x] library(sdR) загружается, sd_system_info() работает
- [x] pipeline.R работает с реальной моделью SD 1.5
- [x] XPtr корректно создаётся/уничтожается
- [x] sd_txt2img() — генерация 512x512 за ~7с (Vulkan GPU)
- [x] sd_save_image() — сохранение PNG
- [x] Vulkan бэкенд работает (radv, AMD GPU)

## Осталось

### 1. [x] Вынести vocab*.hpp из пакета (128 МБ) — БЛОКЕР для CRAN
- Реализовано: скачивание в configure из GitHub Releases
- URL: https://github.com/Zabis13/sdR/releases/tag/assets
- Файлы убраны из git (`.git/info/exclude`), при сборке доступны devtools
- configure скачивает только если файлов нет на диске

### 2. [ ] Уменьшить размер пакета до CRAN лимита (5 МБ)
- Проверить размер без vocab файлов
- Убедиться что остальные исходники (ggml, sd.cpp) укладываются

### 3. [x] Настроить configure скрипт для скачивания зависимостей
- Реализовано в configure: curl/wget, проверка наличия, fallback с инструкцией

### 4. [ ] Подготовить DESCRIPTION и документацию для CRAN
- License, SystemRequirements, URL, BugReports
- Документация всех экспортируемых функций


### 5. [ ] R CMD check --as-cran
- Зависит от задач 1–4
- 0 errors, 0 warnings, 0 notes

### 6. [ ] Предупреждения компиляции (косметика)
- GGML_ATTRIBUTE_FORMAT redefined — r_ggml_compat.h vs ggml.h
- SSE/AVX = 0 в sd_system_info — ggml SIMD определяется при сборке ggmlR, не sdR

### 7. [ ] Доработать функционал
- Проверить img2img, upscaler
- Конвертация изображений (sd_image_t ↔ R raw vector)

### 8. High-res генерация: tiled VAE → tiled pipeline → tiled sampling

Три этапа, каждый строится поверх предыдущего.

#### 8a. [ ] Честный streaming tiled VAE (encode/decode)
**Цель:** encode/decode не держат весь латент/картинку целиком в памяти.

**C++ (src/sd/):**
- Аудит `decode_first_stage()` и `vae_encode()`: найти и убрать скрытые full-frame буферы
- Свести VAE к паттерну:
  1. Вырезать один тайл латента/изображения
  2. Прогнать через VAE (CPU/GPU)
  3. Сразу записать в результирующий буфер с блендингом перекрытий
  4. Перейти к следующему тайлу
- Блендинг: линейные весовые маски по краям, использует только текущий тайл + уже построенную область выхода
- Проверить `sd_tiling_non_square()` в `ggml_extend.hpp` — нет ли там промежуточного полного буфера

**R (pipeline.R):**
- Пробросить все `sd_tiling_params_t` и offload-флаги через Rcpp (частично сделано)
- Добавить `vae_mode = c("normal", "tiled")` с документацией ограничений
- Тесты: decode 4096x4096 латента не должен падать по VRAM

#### 8b. [ ] High-res pipeline без tiled sampling (промежуточный путь)
**Цель:** генерация больших картинок (2K, 4K) уже сейчас, без tiled UNet.

**Стратегия: генерация патчей → сшивка → tiled VAE decode:**
1. Разбить целевое изображение на сетку патчей (напр. 512x512 или 1024x1024)
2. Для каждого патча: обычный `sd_txt2img()` в безопасном разрешении
3. Опционально: `sd_img2img()` с низкой strength для гармонизации стыков
4. Сшить пиксели на R-уровне (панорама, квадратная сетка)
5. Финальный decode/encode через tiled VAE — не ограничивает по разрешению

**R API:**
```r
sd_txt2img_highres(
  ctx, prompt,
  width = 2048, height = 2048,  # целевой размер
  tile = 512,                    # размер патча для UNet
  overlap = 64,                  # перекрытие патчей
  upscale_factor = NULL,         # опциональный ESRGAN апскейл
  img2img_strength = 0.3,        # strength для гармонизации
  vae_tiling = TRUE,             # tiled VAE для encode/decode
  ...
)
```

**Альтернативный простой путь (без сшивки):**
- `sd_txt2img()` на 512x512 → ESRGAN upscale до 2048x2048 → `sd_img2img()` с tiled VAE

#### 8c. [ ] Настоящий tiled sampling (MultiDiffusion-подход)
**Цель:** UNet обрабатывает латент по тайлам, VRAM зависит только от размера тайла.

**C++ API:**
```cpp
// Новая функция в stable-diffusion.cpp
ggml_tensor* sample_tiled(
    ggml_context* work_ctx,
    ggml_tensor* init_latent,      // глобальное латентное полотно (логически большое)
    int tile_size,                  // размер тайла в латентном пространстве (напр. 64 = 512px)
    float tile_overlap,             // перекрытие (0.0-0.5)
    SDCondition cond, SDCondition uncond,
    sample_method_t method,
    const std::vector<float>& sigmas,
    sd_guidance_params_t guidance
);
```

**Алгоритм (на каждом шаге деноизинга):**
1. Создать глобальный `noise_pred` буфер (нули) + `weight_map` (нули)
2. Для каждого тайла в сетке:
   - Вырезать подлатент из глобального `x_t`
   - Запустить UNet только на этот тайл (фиксированный размер → фиксированный VRAM)
   - Умножить на весовую маску (Гаусс или линейная от центра тайла)
   - Прибавить в `noise_pred` и `weight_map`
3. `noise_pred /= weight_map` — нормализация
4. Применить scheduler step: `x_t → x_{t-1}`

**Блендинг тайлов:**
- Перекрытие: 25-50% от размера тайла
- Весовая маска: Гаусс или линейная, максимум в центре тайла, плавное затухание к краям
- Идеи из MultiDiffusion / Tiled Diffusion / Ultimate SD Upscaler

**R API:**
```r
sd_txt2img_tiled(
  ctx, prompt,
  width = 2048, height = 2048,
  tile_size = 64,        # латентных пикселей (= 512 реальных для SD1.5)
  tile_overlap = 0.25,
  vae_tiling = TRUE,     # обязательно для больших размеров
  ...
)
```

**Зависимости:** требует работающий streaming tiled VAE (8a)

#### Порядок реализации
1. **8a** — честный tiled VAE (сейчас, C++ аудит + R проброс)
2. **8b** — high-res pipeline на R-уровне (сразу после 8a, без C++ изменений)
3. **8c** — tiled sampling (крупный эпик, C++ рефакторинг семплера)

### 9. [ ] Multi-GPU inference
- Один sd_ctx работает с одной GPU (выбор через env `SD_VK_DEVICE`)
- Стратегия: несколько процессов, по контексту на GPU, без склейки VRAM
- Каждая GPU держит полную копию модели (~2-3 ГБ для SD 1.5)

**R API:**
```r
sd_txt2img_multi_gpu(
  model_path,          # путь к модели
  prompts,             # вектор промптов
  devices = 0:1,       # индексы GPU
  seeds = NULL,        # сиды (по умолчанию случайные)
  ...                  # остальные параметры sd_txt2img
)
```

**Реализация (только R-слой, C++ не трогаем):**
1. Разбить prompts/seeds на N частей по числу devices
2. Запустить N процессов, каждый с `Sys.setenv(SD_VK_DEVICE = i)`
3. В каждом процессе: `sd_ctx()` → `sd_txt2img()` → вернуть результат
4. Собрать результаты в один список

**Параллелизм:**
- Linux/macOS: `parallel::mclapply()` (forking)
- Кроссплатформенно: `callr::r_bg()` или `future`

**Открытые вопросы:**
- Проверить какая env-переменная реально работает (`SD_VK_DEVICE` / `GGML_VK_DEVICE`)
- Масштабирование: линейное при хорошем I/O, bottleneck — загрузка модели на каждую GPU


