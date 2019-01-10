public static ArrayList<String> foo(String str1) {
		String str2 = str1.trim();
		return Stream.of(str2.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+"))
				.filter(s -> s.length() > 0).map(s -> Common.normalizeName(s, Common.EmptyString))
				.filter(s -> s.length() > 0).collect(Collectors.toCollection(ArrayList::new));
	}