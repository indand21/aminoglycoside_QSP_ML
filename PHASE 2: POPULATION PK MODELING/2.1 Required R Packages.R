# Population PK modeling packages
popPK_packages <- c(
  "nlmixr2",        # Population PK/PD modeling (replaces nlmixr)
  "mrgsolve",       # ODE-based PK/PD simulation
  "rxode2",         # Fast ODE solving (nlmixr2 backend)
  "PKNCA",          # NCA calculations
  "vpc",            # Visual predictive checks
  "xpose",          # Model diagnostics
  "ggPMX"           # Model diagnostics visualization
)

lapply(popPK_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
})
