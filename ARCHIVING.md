# Versioned releases and archival DOI

Canonical repository: https://github.com/jkalya/gatebleed  
Associated paper: https://doi.org/10.1145/3725843.3756097

No archived software DOI is asserted by this file. The DOI in CITATION.cff is
nested under `preferred-citation` and identifies the paper.

## Maintainer release checklist

1. Merge the citation and setup documentation into the canonical repository.
2. Confirm the software contributors in CITATION.cff against the full history;
   the preferred paper citation lists all paper authors in published order.
3. Preserve the existing Apache-2.0 license and notices for bundled third-party components.
4. Validate installation on the supported hardware/environment and record the
   result, dependency versions, required input data, and known limitations in
   release notes. Review notebook outputs and configuration before archiving.
5. In the repository owner's Zenodo account, enable the canonical GitHub
   repository for archiving. This requires suitable GitHub access. If it is
   already linked to a Zenodo record, continue that record's version history.
6. Choose an unused release tag, then set `version` and `date-released` at the
   top level of CITATION.cff to that actual release version/date. Commit them
   before tagging. Do not assign a release number solely to documentation that
   has not been accepted by the maintainers.
7. Publish the GitHub release for the selected commit. Include a concise change
   log, tested setup and reproduction limitations. Wait for Zenodo to process it.
8. Verify that the Zenodo record contains the expected source archive, authors,
   license, version and related paper DOI. Link the paper using `isSupplementTo`
   where supported. Record the version DOI for this exact snapshot and the
   concept DOI for the series of versions.
9. Add the verified software DOI and a DOI badge to the README. If adding a
   top-level `doi` to CITATION.cff, use the software DOI, never the paper DOI.
   Keep the paper DOI under `preferred-citation`. Follow the Zenodo versioning
   workflow for later releases rather than making unrelated duplicate records.

This checklist does not create or archive a release. A contribution fork is not
the canonical project archive.

## Official guidance

- [GitHub citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
- [Enable a repository in Zenodo](https://help.zenodo.org/docs/github/enable-repository/)
- [Archive a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/)

CITATION.cff is the metadata source here. Do not add a competing .zenodo.json
without keeping it synchronized: Zenodo gives .zenodo.json precedence when both
are present.
