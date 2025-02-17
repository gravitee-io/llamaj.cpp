package io.gravitee.llama.cpp;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum AttentionType {
    UNSPECIFIED,
    CAUSAL,
    NON_CAUSAL;

    public static AttentionType fromOrdinal(int ordinal) {
        return values()[ordinal];
    }
}
