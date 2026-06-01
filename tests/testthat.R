library(testthat)
library(sd2R)

# Heavy test files load a real diffusion model (sd_ctx) and run generation,
# which is too slow / resource-hungry for CRAN. They already self-skip when
# SD2R_TEST_MODEL is unset, but we also exclude them entirely on CRAN so no
# model-dependent file runs there. Run them locally with NOT_CRAN=true (and
# SD2R_TEST_MODEL set). Only these two files call sd_ctx(); the rest are pure
# unit tests and run on CRAN.
heavy <- c(
  "lowlevel",
  "preview"
)

on_cran <- !identical(Sys.getenv("NOT_CRAN"), "true")

test_dir <- if (dir.exists("testthat")) "testthat" else "tests/testthat"

if (on_cran) {
  message("--- RUNNING LIGHT TESTS ONLY (set NOT_CRAN=true for the full suite) ---")

  all_tests <- list.files(test_dir, pattern = "^test-.*\\.R$")
  all_names <- sub("^test-(.*)\\.R$", "\\1", all_tests)
  light_tests <- setdiff(all_names, heavy)
  message("Tests to run: ", paste(light_tests, collapse = ", "))

  if (length(light_tests) == 0) {
    test_check("sd2R")
  } else {
    # testthat applies `filter` as grepl(filter, <test name>) with no anchors,
    # so anchor each name with ^...$ to avoid partial-name collisions.
    filter_regex <- paste0("^(", paste(light_tests, collapse = "|"), ")$")
    test_check("sd2R", filter = filter_regex)
  }
} else {
  test_check("sd2R")
}
