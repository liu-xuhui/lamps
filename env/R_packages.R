required_pkgs <- c(
  "RhpcBLASctl",
  "here",
  "earth",
  "stringr",
  "SAM",
  "future",
  "future.apply",
  "hdi"
)

missing <- setdiff(required_pkgs, rownames(installed.packages()))

if (length(missing) > 0) {
  message("Installing missing packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  message("All required packages are already installed.")
}