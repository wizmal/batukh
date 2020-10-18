# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = ./source
BUILDDIR      = docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@if [ -d "$(BUILDDIR)/html" ]; then \
		mv  "$(BUILDDIR)/html/"* "$(BUILDDIR)"; \
		mv  "$(BUILDDIR)/html/".* "$(BUILDDIR)"; \
		rm -r "$(BUILDDIR)/html"; \
	fi
	@if [ -d "$(BUILDDIR)/doctrees" ]; then \
		rm -r "$(BUILDDIR)/doctrees"; \
	fi
	@touch "$(BUILDDIR)/.nojekyll"

# echo $(SPHINXOPTS)
# rm -r "$(BUILDDIR)/doctrees"
# mv "$(BUILDDIR)/html/"* "$(BUILDDIR)"
# mv "$(BUILDDIR)/html/".* "$(BUILDDIR)"
# rmdir "$(BUILDDIR)/html"
