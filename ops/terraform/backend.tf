terraform {
  backend "swift" {
    container         = "lnq-chronicler-tfstate"
    archive_container = "lnq-chronicler-tfstate-archive"
    state_name        = "terraform.tfstate"
  }
}
