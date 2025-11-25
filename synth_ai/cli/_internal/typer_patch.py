from click import Parameter


def patch_typer_make_metavar() -> None:
    try:
        from typer.main import TyperArgument, TyperOption  # type: ignore[import-untyped]
    except Exception:
        return

    def _call_get_metavar(param_type: object, param: Parameter) -> str | None:
        getter = getattr(param_type, "get_metavar", None)
        if getter is None:
            return None
        try:
            return getter(param)
        except TypeError:
            try:
                return getter(param, None)
            except TypeError:
                return None

    def _patched_argument_make_metavar(self, ctx=None) -> str:
        if self.metavar is not None:
            return self.metavar
        var = (self.name or "").upper()
        if not self.required:
            var = f"[{var}]"
        type_var = _call_get_metavar(self.type, self)
        if type_var:
            var += f":{type_var}"
        if self.nargs != 1:
            var += "..."
        return var

    def _patched_option_make_metavar(self, ctx=None) -> str:
        if self.metavar is not None:
            return self.metavar
        metavar = _call_get_metavar(self.type, self)
        if not metavar:
            name = getattr(self.type, "name", "") or ""
            metavar = name.upper()
        if self.nargs != 1:
            metavar += "..."
        return metavar

    TyperArgument.make_metavar = _patched_argument_make_metavar  # type: ignore[assignment]
    TyperOption.make_metavar = _patched_option_make_metavar  # type: ignore[assignment]
