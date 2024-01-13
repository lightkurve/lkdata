def test_module_imports():
    """
    Test that modules can be imported as expected in the new structure.
    """
    import lightkurve  # noqa: F401
    from lightkurve.missions.kepler import KeplerQualityFlags  # noqa: F401
    from lightkurve.missions.tess import TESSQualityFlags  # noqa: F401

    #    from lightkurve.correctors import design_matrix  # noqa: F401
